"""Unit 1 (R3/R6): client ``https://`` endpoint support with fail-closed
certificate verification.

The coordinator client can speak *verified* TLS to a terminating front. The
loopback ``http`` path stays byte-identical to today (regression matrix lives in
``tests/test_coherence_client_remote_endpoint.py`` and is untouched here).

Test-first ordering: these scenarios were written before the implementation and
watched to fail (missing scheme/context/redirect wiring), then the code was
added to make them pass.

Two families:

- **Pure-function tests** — ``base_url`` scheme rendering, the SSL-context
  factory invariants, CA-file discipline, ``from_env`` plumbing. No sockets;
  always run.
- **Socket tests** — an in-process ``ThreadingHTTPServer`` wrapped in a
  server-side ``SSLContext``, fronted by a throwaway CA + IP-SAN server cert
  minted with the ``openssl`` CLI. These ``pytest.skip`` when ``openssl`` is
  not on PATH so the pure-function coverage still runs everywhere.

The cert profile deliberately satisfies Python 3.13's ``VERIFY_X509_STRICT``
(the default under ``create_default_context``): subjectKeyIdentifier,
authorityKeyIdentifier, critical basicConstraints, keyUsage,
extendedKeyUsage=serverAuth, and ``subjectAltName=IP:127.0.0.1`` (an IP SAN,
because our endpoints are IP literals and OpenSSL matches IP SANs natively).
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from ccs.cli import _coherence_client as cc
from ccs.cli._coherence_client import (
    CoordinatorEndpoint,
    RemoteCoordinatorConfig,
    build_tls_context,
    resolve_remote_endpoint,
)
from ccs.core.exceptions import (
    InsecureTransportRefused,
    RedirectRefused,
    TlsConfigError,
    TlsVerificationFailed,
)

_OPENSSL = shutil.which("openssl")
requires_openssl = pytest.mark.skipif(
    _OPENSSL is None, reason="openssl CLI not on PATH (socket TLS tests skipped)"
)


# ===========================================================================
# Pure-function tests (no sockets) — always run.
# ===========================================================================


class TestBaseUrlScheme:
    def test_http_scheme_is_the_default(self) -> None:
        ep = CoordinatorEndpoint(port=8080, bearer="deadbeef", host="10.0.0.5")
        assert ep.scheme == "http"
        assert ep.base_url == "http://10.0.0.5:8080"

    def test_https_scheme_renders_https(self) -> None:
        ep = CoordinatorEndpoint(
            port=8443, bearer="deadbeef", host="10.0.0.5", scheme="https"
        )
        assert ep.base_url == "https://10.0.0.5:8443"

    def test_https_scheme_brackets_ipv6(self) -> None:
        ep = CoordinatorEndpoint(
            port=8443, bearer="deadbeef", host="fd00::1", scheme="https"
        )
        assert ep.base_url == "https://[fd00::1]:8443"

    def test_http_scheme_still_brackets_ipv6(self) -> None:
        # Regression: the IPv6-bracketing branch must fire for BOTH schemes.
        ep = CoordinatorEndpoint(port=8080, bearer="deadbeef", host="::1")
        assert ep.base_url == "http://[::1]:8080"


class TestTlsContextFactory:
    def test_default_context_enforces_verification(self) -> None:
        ctx = build_tls_context()
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_default_context_disables_legacy_cn_fallback(self) -> None:
        # SAN-only tightening: the legacy Common Name fallback is off, so a cert
        # with no matching SAN is rejected even if its CN matches (this is what
        # makes the DNS-only-SAN-on-IP-endpoint case fail closed).
        ctx = build_tls_context()
        assert getattr(ctx, "hostname_checks_common_name", False) is False

    def test_default_context_pins_tls12_floor(self) -> None:
        ctx = build_tls_context()
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_no_off_switch_is_representable(self) -> None:
        # There is intentionally no parameter that could yield CERT_NONE or
        # check_hostname=False — the factory takes only an optional CA path.
        ctx = build_tls_context(ca_file=None)
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_missing_ca_file_raises_tls_config_error_naming_path(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "nope-ca.pem"
        with pytest.raises(TlsConfigError) as exc:
            build_tls_context(ca_file=str(missing))
        assert str(missing) in str(exc.value)

    def test_non_pem_ca_file_raises_tls_config_error_naming_path(
        self, tmp_path: Path
    ) -> None:
        garbage = tmp_path / "garbage-ca.pem"
        garbage.write_text("this is not a certificate", encoding="utf-8")
        with pytest.raises(TlsConfigError) as exc:
            build_tls_context(ca_file=str(garbage))
        assert str(garbage) in str(exc.value)

    def test_non_utf8_ca_file_raises_tls_config_error_naming_path(
        self, tmp_path: Path
    ) -> None:
        # Non-UTF-8 bytes exercise the UnicodeDecodeError branch in the CA reader
        # (distinct from the valid-UTF-8-but-not-PEM SSLError branch above); it
        # must fail closed as a typed TlsConfigError, never a raw UnicodeDecodeError.
        binary = tmp_path / "binary-ca.pem"
        binary.write_bytes(b"\xff\xfe\x00\x01not a cert")
        with pytest.raises(TlsConfigError) as exc:
            build_tls_context(ca_file=str(binary))
        assert str(binary) in str(exc.value)

    def test_symlinked_ca_file_is_refused(self, tmp_path: Path) -> None:
        real = tmp_path / "real-ca.pem"
        real.write_text("dummy", encoding="utf-8")
        link = tmp_path / "link-ca.pem"
        link.symlink_to(real)
        with pytest.raises(TlsConfigError) as exc:
            build_tls_context(ca_file=str(link))
        assert str(link) in str(exc.value)

    def test_group_writable_ca_file_is_refused(self, tmp_path: Path) -> None:
        import os

        ca = tmp_path / "loose-ca.pem"
        ca.write_text("dummy", encoding="utf-8")
        os.chmod(ca, 0o664)  # group-writable — the attack surface for a trust anchor
        with pytest.raises(TlsConfigError) as exc:
            build_tls_context(ca_file=str(ca))
        assert str(ca) in str(exc.value)

    def test_world_writable_ca_file_is_refused(self, tmp_path: Path) -> None:
        import os

        ca = tmp_path / "world-ca.pem"
        ca.write_text("dummy", encoding="utf-8")
        os.chmod(ca, 0o666)
        with pytest.raises(TlsConfigError):
            build_tls_context(ca_file=str(ca))

    def test_readable_but_not_writable_ca_file_is_accepted(
        self, tmp_path: Path
    ) -> None:
        # Certs are public: group/world-READABLE is fine (0o644). Only the
        # WRITABLE bits are the attack (a swapped trust anchor).
        import os

        real_ca = _make_ca_pem(tmp_path)
        if real_ca is None:
            pytest.skip("openssl not available to mint a valid PEM")
        os.chmod(real_ca, 0o644)
        ctx = build_tls_context(ca_file=str(real_ca))
        assert ctx.verify_mode == ssl.CERT_REQUIRED


class TestFromEnvTlsPlumbing:
    def test_ccs_remote_tls_selects_https_scheme(self) -> None:
        cfg = RemoteCoordinatorConfig.from_env(
            env={"CCS_REMOTE_COORDINATOR": "1", "CCS_REMOTE_TLS": "1"}
        )
        assert cfg.scheme == "https"

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_ccs_remote_tls_truthy_values(self, val: str) -> None:
        cfg = RemoteCoordinatorConfig.from_env(
            env={"CCS_REMOTE_COORDINATOR": "1", "CCS_REMOTE_TLS": val}
        )
        assert cfg.scheme == "https"

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "  "])
    def test_ccs_remote_tls_falsey_stays_http(self, val: str) -> None:
        cfg = RemoteCoordinatorConfig.from_env(
            env={"CCS_REMOTE_COORDINATOR": "1", "CCS_REMOTE_TLS": val}
        )
        assert cfg.scheme == "http"

    def test_default_scheme_is_http(self) -> None:
        cfg = RemoteCoordinatorConfig.from_env(env={"CCS_REMOTE_COORDINATOR": "1"})
        assert cfg.scheme == "http"
        assert cfg.ca_file is None

    def test_ca_file_path_is_carried(self, tmp_path: Path) -> None:
        # File-path not inline (mirrors CCS_REMOTE_SECRET_FILE): the path is
        # carried verbatim; existence/permission validation happens at factory
        # time, not parse time.
        ca_path = tmp_path / "ca.pem"
        cfg = RemoteCoordinatorConfig.from_env(
            env={
                "CCS_REMOTE_COORDINATOR": "1",
                "CCS_REMOTE_TLS": "1",
                "CCS_REMOTE_CA_FILE": str(ca_path),
            }
        )
        assert cfg.ca_file == str(ca_path)

    def test_resolve_remote_endpoint_threads_scheme_and_ca(self) -> None:
        ep = resolve_remote_endpoint(
            "10.0.0.5",
            8443,
            "s3cr3t",
            scheme="https",
            env={"CCS_REMOTE_INSECURE": "1"},
        )
        assert ep.scheme == "https"
        assert ep.base_url == "https://10.0.0.5:8443"

    def test_resolve_remote_endpoint_defaults_preserve_http(self) -> None:
        ep = resolve_remote_endpoint(
            "10.0.0.5", 8080, "s3cr3t", env={"CCS_REMOTE_INSECURE": "1"}
        )
        assert ep.scheme == "http"
        assert ep.base_url == "http://10.0.0.5:8080"


# ===========================================================================
# Socket tests — real TLS handshake against an in-process server.
# ===========================================================================


def _run_openssl(args: list[str], cwd: Path) -> None:
    subprocess.run(
        [_OPENSSL, *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _make_ca_pem(tmp_path: Path) -> Path | None:
    """Mint just a CA cert PEM (for the readable-CA-file acceptance test)."""
    if _OPENSSL is None:
        return None
    tmp_path.mkdir(parents=True, exist_ok=True)
    ca_cnf = tmp_path / "ca.cnf"
    ca_cnf.write_text(
        "[req]\ndistinguished_name=dn\nx509_extensions=v3_ca\nprompt=no\n"
        "[dn]\nCN=coherence-test-ca\n"
        "[v3_ca]\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n"
        "basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\n",
        encoding="utf-8",
    )
    ca_key = tmp_path / "ca.key"
    ca_pem = tmp_path / "ca.pem"
    _run_openssl(
        [
            "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(ca_key), "-out", str(ca_pem),
            "-days", "1", "-nodes", "-config", str(ca_cnf),
        ],
        tmp_path,
    )
    return ca_pem


@dataclass(frozen=True)
class _CertBundle:
    ca_pem: Path
    server_cert: Path
    server_key: Path


def _sign_leaf(
    tmp_path: Path, ca_pem: Path, ca_key: Path, san_line: str, name: str
) -> tuple[Path, Path]:
    """Sign an RFC5280-strict leaf cert with the given subjectAltName line."""
    leaf_cnf = tmp_path / f"{name}.cnf"
    leaf_cnf.write_text(
        "[req]\ndistinguished_name=dn\nprompt=no\n[dn]\nCN=coherence-test-server\n",
        encoding="utf-8",
    )
    ext_cnf = tmp_path / f"{name}_ext.cnf"
    ext_cnf.write_text(
        "subjectKeyIdentifier=hash\n"
        "authorityKeyIdentifier=keyid:always\n"
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        f"subjectAltName={san_line}\n",
        encoding="utf-8",
    )
    key = tmp_path / f"{name}.key"
    csr = tmp_path / f"{name}.csr"
    cert = tmp_path / f"{name}.pem"
    _run_openssl(
        ["req", "-newkey", "rsa:2048", "-keyout", str(key), "-out", str(csr),
         "-nodes", "-config", str(leaf_cnf)],
        tmp_path,
    )
    _run_openssl(
        ["x509", "-req", "-in", str(csr), "-CA", str(ca_pem), "-CAkey", str(ca_key),
         "-CAcreateserial", "-out", str(cert), "-days", "1", "-extfile", str(ext_cnf)],
        tmp_path,
    )
    return cert, key


def _mint_bundle(tmp_path: Path, san_line: str = "IP:127.0.0.1") -> _CertBundle:
    ca_pem = _make_ca_pem(tmp_path)
    assert ca_pem is not None
    ca_key = tmp_path / "ca.key"
    cert, key = _sign_leaf(tmp_path, ca_pem, ca_key, san_line, "server")
    return _CertBundle(ca_pem=ca_pem, server_cert=cert, server_key=key)


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    #: Populated per-server-instance; each item is the Authorization header a
    #: request presented (or None). Non-empty => the bearer reached this server.
    seen_authorizations: list[str | None] = []
    #: Set on the *first* server to force a 302 to this location (redirect test).
    redirect_to: str | None = None

    def _record_and_respond(self) -> None:
        type(self).seen_authorizations.append(self.headers.get("Authorization"))
        if type(self).redirect_to is not None:
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def do_GET(self) -> None:  # noqa: N802 (stdlib handler contract)
        self._record_and_respond()

    def do_POST(self) -> None:  # noqa: N802 (stdlib handler contract)
        # Drain the body before answering so the client never sees a reset.
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._record_and_respond()

    def log_message(self, *args: object) -> None:  # silence test noise
        pass


def _make_handler_class(
    redirect_to: str | None = None,
) -> type[_RecordingHandler]:
    # A fresh subclass per server so seen_authorizations/redirect_to don't leak
    # across the two servers in the redirect test.
    return type(
        "_ScopedHandler",
        (_RecordingHandler,),
        {"seen_authorizations": [], "redirect_to": redirect_to},
    )


@dataclass
class _RunningServer:
    port: int
    handler_cls: type[_RecordingHandler]
    _server: http.server.ThreadingHTTPServer

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _start_tls_server(
    bundle: _CertBundle, handler_cls: type[_RecordingHandler]
) -> _RunningServer:
    sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sctx.load_cert_chain(str(bundle.server_cert), str(bundle.server_key))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv.socket = sctx.wrap_socket(srv.socket, server_side=True)
    port = srv.socket.getsockname()[1]
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return _RunningServer(port=port, handler_cls=handler_cls, _server=srv)


def _start_plain_server(handler_cls: type[_RecordingHandler]) -> _RunningServer:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = srv.socket.getsockname()[1]
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return _RunningServer(port=port, handler_cls=handler_cls, _server=srv)


@pytest.fixture
def tls_bundle(tmp_path: Path) -> _CertBundle:
    if _OPENSSL is None:
        pytest.skip("openssl CLI not on PATH")
    return _mint_bundle(tmp_path)


@requires_openssl
class TestHttpsRequestPath:
    def test_https_request_with_matching_ip_san_cert_succeeds(
        self, tls_bundle: _CertBundle
    ) -> None:
        handler = _make_handler_class()
        srv = _start_tls_server(tls_bundle, handler)
        try:
            ep = CoordinatorEndpoint(
                port=srv.port,
                bearer="s3cr3t",
                host="127.0.0.1",
                scheme="https",
                ca_file=str(tls_bundle.ca_pem),
            )
            body = cc.get(ep, "/status")
            assert body == {"ok": True}
            # The bearer DID ride the verified hop (positive control).
            assert srv.handler_cls.seen_authorizations == ["Bearer s3cr3t"]
        finally:
            srv.shutdown()

    def test_private_ca_bundle_validates_server(
        self, tls_bundle: _CertBundle
    ) -> None:
        # Same happy path, framed as the CCS_REMOTE_CA_FILE story: an exclusive
        # private-CA bundle validates a private-CA-signed server.
        handler = _make_handler_class()
        srv = _start_tls_server(tls_bundle, handler)
        try:
            ep = CoordinatorEndpoint(
                port=srv.port,
                bearer="s3cr3t",
                host="127.0.0.1",
                scheme="https",
                ca_file=str(tls_bundle.ca_pem),
            )
            assert cc.get(ep, "/status") == {"ok": True}
        finally:
            srv.shutdown()

    def test_cert_not_signed_by_trusted_ca_refuses_and_bearer_never_sent(
        self, tmp_path: Path
    ) -> None:
        # Server presents a cert from CA #1; the client trusts a DIFFERENT CA #2.
        server_bundle = _mint_bundle(tmp_path / "srv")
        client_only = tmp_path / "cli"
        client_only.mkdir()
        other_ca = _make_ca_pem(client_only)
        assert other_ca is not None

        handler = _make_handler_class()
        srv = _start_tls_server(server_bundle, handler)
        try:
            ep = CoordinatorEndpoint(
                port=srv.port,
                bearer="s3cr3t",
                host="127.0.0.1",
                scheme="https",
                ca_file=str(other_ca),
            )
            with pytest.raises(TlsVerificationFailed) as exc:
                cc.get(ep, "/status")
            assert exc.value.host == "127.0.0.1"
            # The handshake failed BEFORE any HTTP request — the bearer never
            # reached the server.
            assert srv.handler_cls.seen_authorizations == []
        finally:
            srv.shutdown()

    def test_dns_only_san_cert_on_ip_endpoint_fails_closed(
        self, tmp_path: Path
    ) -> None:
        # Pins the OpenSSL behavior we rely on: an IP-literal endpoint against a
        # DNS-only-SAN cert must NOT validate (no CN fallback, no name match).
        bundle = _mint_bundle(tmp_path, san_line="DNS:coherence.example")
        handler = _make_handler_class()
        srv = _start_tls_server(bundle, handler)
        try:
            ep = CoordinatorEndpoint(
                port=srv.port,
                bearer="s3cr3t",
                host="127.0.0.1",
                scheme="https",
                ca_file=str(bundle.ca_pem),
            )
            with pytest.raises(TlsVerificationFailed):
                cc.get(ep, "/status")
            assert srv.handler_cls.seen_authorizations == []
        finally:
            srv.shutdown()


@requires_openssl
class TestRedirectRefusal:
    def test_any_3xx_refused_and_target_receives_no_request(
        self, tmp_path: Path
    ) -> None:
        # Two TLS servers sharing one CA: the first 302s to the second. A bare
        # urlopen would follow AND copy Authorization onto the second hop before
        # returning — so we assert the target saw NO request at all.
        bundle = _mint_bundle(tmp_path)
        target_handler = _make_handler_class()
        target = _start_tls_server(bundle, target_handler)
        redirecting_handler = _make_handler_class(
            redirect_to=f"https://127.0.0.1:{target.port}/elsewhere"
        )
        redirecting = _start_tls_server(bundle, redirecting_handler)
        try:
            ep = CoordinatorEndpoint(
                port=redirecting.port,
                bearer="s3cr3t",
                host="127.0.0.1",
                scheme="https",
                ca_file=str(bundle.ca_pem),
            )
            with pytest.raises(RedirectRefused) as exc:
                cc.get(ep, "/status")
            # The refusal names the status, never the attempted location:
            # the Location is the redirector's text, and may echo what it
            # was sent.
            assert exc.value.status == 302
            assert exc.value.location == cc.REDIRECT_LOCATION_WITHHELD
            assert "elsewhere" not in str(exc.value)
            # The redirect TARGET must have received nothing — the bearer never
            # rode the hop.
            assert target.handler_cls.seen_authorizations == []
        finally:
            redirecting.shutdown()
            target.shutdown()

    @pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
    def test_http_endpoint_3xx_is_also_refused(self, code: int) -> None:
        # Redirect refusal is scheme-agnostic AND uniform across 3xx codes: any
        # 3xx from a plaintext http endpoint is refused (the coordinator is one
        # fixed endpoint). 308 in particular has no stdlib redirect method — the
        # _NoRedirectHandler.http_error_308 alias makes it a typed refusal too.
        class _PlainHandler(http.server.BaseHTTPRequestHandler):
            seen: list[str | None] = []
            status = 302

            def do_GET(self) -> None:  # noqa: N802
                type(self).seen.append(self.headers.get("Authorization"))
                self.send_response(type(self).status)
                self.send_header("Location", "http://127.0.0.1:1/elsewhere")
                self.end_headers()

            def log_message(self, *a: object) -> None:
                pass

        scoped = type("_ScopedPlain", (_PlainHandler,), {"seen": [], "status": code})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), scoped)
        port = srv.socket.getsockname()[1]
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        try:
            ep = CoordinatorEndpoint(port=port, bearer="s3cr3t", host="127.0.0.1")
            with pytest.raises(RedirectRefused):
                cc.get(ep, "/status")
        finally:
            srv.shutdown()
            srv.server_close()


@requires_openssl
class TestFromEnvIntegration:
    def test_from_env_https_with_ca_mints_working_endpoint(
        self, tls_bundle: _CertBundle
    ) -> None:
        handler = _make_handler_class()
        srv = _start_tls_server(tls_bundle, handler)
        try:
            cfg = RemoteCoordinatorConfig.from_env(
                env={
                    "CCS_REMOTE_COORDINATOR": "1",
                    "CCS_REMOTE_TLS": "1",
                    "CCS_REMOTE_HOST": "127.0.0.1",
                    "CCS_REMOTE_PORT": str(srv.port),
                    "CCS_REMOTE_CA_FILE": str(tls_bundle.ca_pem),
                }
            )
            assert cfg.scheme == "https"
            assert cfg.ca_file == str(tls_bundle.ca_pem)
            ep = resolve_remote_endpoint(
                "127.0.0.1",
                srv.port,
                "s3cr3t",
                scheme=cfg.scheme,
                ca_file=cfg.ca_file,
                env={"CCS_REMOTE_INSECURE": "1"},
            )
            assert ep.base_url == f"https://127.0.0.1:{srv.port}"
            assert cc.get(ep, "/status") == {"ok": True}
            assert srv.handler_cls.seen_authorizations == ["Bearer s3cr3t"]
        finally:
            srv.shutdown()


# ===========================================================================
# Regression: the cross-host demo's TLS resolution pattern (examples/cross_host/
# main.py) must thread scheme/ca_file from from_env into resolve_remote_endpoint.
# Pure-function (no socket) — pins the exact contract a review caught main.py
# violating: an https routed host needs NO CCS_REMOTE_INSECURE ack.
# ===========================================================================


class TestTlsSatisfiesGuardForRoutedHost:
    def test_from_env_tls_routed_host_resolves_without_insecure_ack(self) -> None:
        cfg = RemoteCoordinatorConfig.from_env(
            env={
                "CCS_REMOTE_COORDINATOR": "1",
                "CCS_REMOTE_TLS": "1",
                "CCS_REMOTE_HOST": "10.0.0.5",
                "CCS_REMOTE_PORT": "8443",
            }
        )
        assert cfg.scheme == "https"
        # Threaded (what main.py now does): the verified-TLS scheme satisfies the
        # guard for a routed host with NO ack — resolves cleanly.
        ep = resolve_remote_endpoint(
            cfg.host, cfg.port, "s3cr3t", scheme=cfg.scheme, ca_file=cfg.ca_file, env={}
        )
        assert ep.base_url == "https://10.0.0.5:8443"
        assert ep.scheme == "https"

    def test_dropping_the_scheme_thread_through_reproduces_the_bug(self) -> None:
        # The signature of the main.py bug: without scheme= the endpoint defaults
        # to http, so a routed host without the ack is refused. This is exactly
        # what a caller following the TLS docs would hit if the thread-through
        # were dropped — kept as a guard against regressing it.
        with pytest.raises(InsecureTransportRefused):
            resolve_remote_endpoint("10.0.0.5", 8443, "s3cr3t", env={})


# ===========================================================================
# Edge: loopback http endpoint — byte-identical behavior to today.
# ===========================================================================


class TestLoopbackHttpUnchanged:
    def test_loopback_http_endpoint_uses_no_tls_context(self, tmp_path: Path) -> None:
        # A plain http loopback server: the request must succeed WITHOUT any TLS
        # machinery (the context param is None on http paths).
        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"loopback": true}')

            def log_message(self, *a: object) -> None:
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        port = srv.socket.getsockname()[1]
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        try:
            ep = CoordinatorEndpoint(port=port, bearer="deadbeef", host="127.0.0.1")
            assert ep.scheme == "http"
            assert cc.get(ep, "/status") == {"loopback": True}
        finally:
            srv.shutdown()
            srv.server_close()


# ===========================================================================
# Opener construction. On Python 3.12+ a default urllib HTTPSHandler loads the
# whole system CA store when it is constructed (~13 ms of CPU), and the
# transport used to build one into a fresh opener on EVERY request, plain http
# to loopback included. Plain http now reuses one opener with no https handler;
# https with no CA file reuses one whose system-trust context is built once per
# process; https with CCS_REMOTE_CA_FILE still builds its verified context per
# request, so that bundle is re-validated every time.
# ===========================================================================


@dataclass
class _TlsSetupCounts:
    https_handlers: int = 0
    ca_store_loads: int = 0


@pytest.fixture
def tls_setup_counts(monkeypatch: pytest.MonkeyPatch) -> _TlsSetupCounts:
    """Count ``HTTPSHandler`` constructions and default CA-store loads.

    ``https_handlers`` catches the regression on every supported Python: the old
    transport built a default ``HTTPSHandler`` per request on 3.11 too, where it
    only happened to be cheap. ``ca_store_loads`` counts the cost itself, which
    3.12+ pays. It patches ``SSLContext.set_default_verify_paths``, not
    ``ssl.create_default_context``: the stdlib calls that function through a
    second name, ``ssl._create_default_https_context``, which a patch on the
    first name never sees.
    """
    counts = _TlsSetupCounts()
    real_init = urllib.request.HTTPSHandler.__init__
    real_load = ssl.SSLContext.set_default_verify_paths

    def counting_init(
        self: urllib.request.HTTPSHandler, *args: object, **kwargs: object
    ) -> None:
        counts.https_handlers += 1
        real_init(self, *args, **kwargs)

    def counting_load(self: ssl.SSLContext) -> None:
        counts.ca_store_loads += 1
        real_load(self)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "__init__", counting_init)
    monkeypatch.setattr(ssl.SSLContext, "set_default_verify_paths", counting_load)
    return counts


@pytest.fixture
def fresh_shared_openers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Shared openers are built once per process. Clear them so the request
    # that builds one runs inside the test.
    monkeypatch.setattr(cc, "_shared_openers", {})


@pytest.fixture
def opener_builds(
    fresh_shared_openers: None, monkeypatch: pytest.MonkeyPatch
) -> list[None]:
    """One entry per ``OpenerDirector`` built during the test."""
    builds: list[None] = []
    real_init = urllib.request.OpenerDirector.__init__

    def recording_init(self: urllib.request.OpenerDirector) -> None:
        builds.append(None)
        real_init(self)

    monkeypatch.setattr(urllib.request.OpenerDirector, "__init__", recording_init)
    return builds


class _EchoHandler(_RecordingHandler):
    """Answers with the path and bearer it received, so a response that belongs
    to another thread's request cannot pass for this one's."""

    def _record_and_respond(self) -> None:
        body = json.dumps(
            {"path": self.path, "authorization": self.headers.get("Authorization")}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _NotFoundHandler(_RecordingHandler):
    """Answers every request with a 404 and a JSON error body."""

    def _record_and_respond(self) -> None:
        type(self).seen_authorizations.append(self.headers.get("Authorization"))
        body = b'{"error": "no such route"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestTlsSetupCountsSeeTheCost:
    """Positive controls: the counters see what the old transport did.

    Without these, a counter that stopped intercepting (say, a renamed stdlib
    internal) would turn the zero-count assertions below green for the wrong
    reason.
    """

    def test_the_old_per_request_opener_is_counted(
        self, tls_setup_counts: _TlsSetupCounts
    ) -> None:
        # The body of the old _build_opener(None): build_opener with only the
        # redirect handler, which adds a default HTTPSHandler on every Python.
        urllib.request.build_opener(cc._NoRedirectHandler())
        assert tls_setup_counts.https_handlers == 1

    @pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="3.11's HTTPSHandler builds its SSL context lazily, at connect time",
    )
    def test_the_old_per_request_opener_loads_the_ca_store(
        self, tls_setup_counts: _TlsSetupCounts
    ) -> None:
        urllib.request.build_opener(cc._NoRedirectHandler())
        assert tls_setup_counts.ca_store_loads == 1

    def test_a_default_context_is_counted(
        self, tls_setup_counts: _TlsSetupCounts
    ) -> None:
        ssl.create_default_context()
        assert tls_setup_counts.ca_store_loads == 1


class TestPlainHttpOpener:
    def test_http_requests_build_no_https_handler_and_load_no_ca_store(
        self, tls_setup_counts: _TlsSetupCounts, fresh_shared_openers: None
    ) -> None:
        srv = _start_plain_server(_make_handler_class())
        try:
            ep = CoordinatorEndpoint(port=srv.port, bearer="s3cr3t", host="127.0.0.1")
            for _ in range(3):
                assert cc.get(ep, "/status") == {"ok": True}
            assert cc.post(ep, "/hooks/pre-read", {"path": "a.md"}) == {"ok": True}
            # All four requests really went out, the first one building the opener.
            assert srv.handler_cls.seen_authorizations == ["Bearer s3cr3t"] * 4
            assert tls_setup_counts.https_handlers == 0
            assert tls_setup_counts.ca_store_loads == 0
        finally:
            srv.shutdown()

    def test_http_requests_reuse_one_opener(self, opener_builds: list[None]) -> None:
        srv = _start_plain_server(_make_handler_class())
        try:
            ep = CoordinatorEndpoint(port=srv.port, bearer="s3cr3t", host="127.0.0.1")
            cc.get(ep, "/status")
            assert len(opener_builds) == 1  # the first request builds it
            for _ in range(4):
                cc.get(ep, "/status")
            cc.post(ep, "/hooks/session-stop", {"session_id": "s1"})
            assert len(opener_builds) == 1  # ...and every later one reuses it
        finally:
            srv.shutdown()

    def test_the_shared_opener_cannot_open_https(
        self, fresh_shared_openers: None
    ) -> None:
        # It holds no https handler, so an https request routed to it fails
        # closed instead of riding a default (possibly unverified) SSL context.
        with pytest.raises(urllib.error.URLError, match="unknown url type: https"):
            cc._get_shared_opener(system_tls=False).open("https://127.0.0.1:1/status")

    def test_an_error_status_raises_http_error_with_its_body(
        self, fresh_shared_openers: None
    ) -> None:
        # The handler list is hand-built, so HTTPDefaultErrorHandler is one entry
        # an edit can drop. Without it a 404 comes back as a None response, and
        # ``with opener.open(...)`` raises a TypeError that no except clause in
        # _execute catches: the console scripts would print a traceback instead
        # of the HTTPError they handle. Every other test here answers 2xx or 3xx.
        handler = type("_Scoped404", (_NotFoundHandler,), {"seen_authorizations": []})
        srv = _start_plain_server(handler)
        try:
            ep = CoordinatorEndpoint(port=srv.port, bearer="s3cr3t", host="127.0.0.1")
            with pytest.raises(urllib.error.HTTPError) as exc:
                cc.get(ep, "/missing")
            assert exc.value.code == 404
            assert cc.http_status_from_error(exc.value) == {"error": "no such route"}
            assert srv.handler_cls.seen_authorizations == ["Bearer s3cr3t"]
        finally:
            srv.shutdown()

    @pytest.mark.parametrize(
        "scheme", ["http", pytest.param("https", marks=requires_openssl)]
    )
    def test_one_opener_serves_concurrent_threads_without_crosstalk(
        self,
        scheme: str,
        opener_builds: list[None],
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # CoherentVolume instances on different threads share this module and so
        # its openers: plain http, and https verified against the system trust
        # store. All threads start at once on an unbuilt opener, then check that
        # every response answers their own request (path and bearer).
        n_threads, n_requests = 8, 20
        if scheme == "https":
            bundle: _CertBundle = request.getfixturevalue("tls_bundle")
            monkeypatch.setenv("SSL_CERT_FILE", str(bundle.ca_pem))
            srv = _start_tls_server(bundle, _EchoHandler)
        else:
            srv = _start_plain_server(_EchoHandler)
        barrier = threading.Barrier(n_threads)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            ep = CoordinatorEndpoint(
                port=srv.port, bearer=f"b{i}", host="127.0.0.1", scheme=scheme
            )
            try:
                barrier.wait(timeout=10)
                for j in range(n_requests):
                    path = f"/echo/{i}/{j}"
                    body = cc.get(ep, path)
                    assert body == {"path": path, "authorization": f"Bearer b{i}"}
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i,), name=f"opener-worker-{i}")
            for i in range(n_threads)
        ]
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # interleave threads inside the first-use build
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
                assert not thread.is_alive(), (
                    f"{thread.name} did not finish {n_requests} requests within 60s"
                )
        finally:
            sys.setswitchinterval(old_interval)
            srv.shutdown()
        assert errors == []
        assert len(opener_builds) == 1


@dataclass
class _ProxyRecorder:
    port: int
    connections: list[None]


@pytest.fixture
def proxy_recorder(monkeypatch: pytest.MonkeyPatch) -> Iterator[_ProxyRecorder]:
    """A listener standing in for a proxy: it records every connection made to it.

    Any connection counts, whether an absolute-URI GET or a CONNECT, so it sees a
    proxied request of either kind. ``no_proxy`` is cleared so nothing exempts a
    host from a proxy that is set.
    """
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    listener = socket.create_server(("127.0.0.1", 0))
    # close() from another thread does not wake a blocked accept() on Linux, so
    # poll with a short timeout and stop on a flag.
    listener.settimeout(0.05)
    stop = threading.Event()
    recorded = _ProxyRecorder(port=listener.getsockname()[1], connections=[])

    def record_connections() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:  # includes socket.timeout
                continue
            recorded.connections.append(None)
            conn.close()

    thread = threading.Thread(target=record_connections, daemon=True)
    thread.start()
    yield recorded
    stop.set()
    thread.join(timeout=5)
    listener.close()


def _assert_a_stock_opener_uses_the_proxy(recorder: _ProxyRecorder, url: str) -> None:
    """Positive control: under the test's proxy settings, urllib's own opener
    does connect to the recorder. Without it, ``connections == []`` after the
    client call could mean the proxy was never set, not that the client skipped
    it. The recorder hangs up, so the stock request fails."""
    with pytest.raises(OSError):
        urllib.request.build_opener().open(url, timeout=5)
    assert recorder.connections == [None]
    recorder.connections.clear()


@pytest.mark.parametrize(
    ("endpoint_scheme", "proxy_scheme"),
    [
        ("http", "http"),
        ("http", "https"),
        pytest.param("https", "http", marks=requires_openssl),
    ],
)
def test_requests_ignore_proxy_settings(
    endpoint_scheme: str,
    proxy_scheme: str,
    proxy_recorder: _ProxyRecorder,
    fresh_shared_openers: None,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No coordinator request goes through a proxy. A shared opener that honoured
    # one would send the bearer there, keep using a proxy captured at first use
    # after it went away, and fail outright on an https:// proxy for plain http.
    monkeypatch.setenv(
        f"{endpoint_scheme}_proxy",
        f"{proxy_scheme}://127.0.0.1:{proxy_recorder.port}",
    )
    if endpoint_scheme == "https":
        bundle: _CertBundle = request.getfixturevalue("tls_bundle")
        monkeypatch.setenv("SSL_CERT_FILE", str(bundle.ca_pem))
        srv = _start_tls_server(bundle, _make_handler_class())
    else:
        srv = _start_plain_server(_make_handler_class())
    try:
        _assert_a_stock_opener_uses_the_proxy(
            proxy_recorder, f"{endpoint_scheme}://127.0.0.1:{srv.port}/status"
        )
        ep = CoordinatorEndpoint(
            port=srv.port, bearer="s3cr3t", host="127.0.0.1", scheme=endpoint_scheme
        )
        assert cc.get(ep, "/status") == {"ok": True}
        assert srv.handler_cls.seen_authorizations == ["Bearer s3cr3t"]
        assert proxy_recorder.connections == []
    finally:
        srv.shutdown()


def test_a_routed_host_ignores_proxy_settings(
    proxy_recorder: _ProxyRecorder,
    fresh_shared_openers: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The no-proxy rule covers remote endpoints too, not only loopback: a proxy
    # that exempted loopback would still receive the bearer for a routed host.
    # 192.0.2.1 (TEST-NET-1) is never routed, so the direct attempt times out.
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy_recorder.port}")
    monkeypatch.setattr(cc, "CLI_HTTP_TIMEOUT_SEC", 0.5)
    _assert_a_stock_opener_uses_the_proxy(proxy_recorder, "http://192.0.2.1:8080/status")
    ep = CoordinatorEndpoint(port=8080, bearer="s3cr3t", host="192.0.2.1")
    with pytest.raises(cc.CoordinatorUnavailable):
        cc.get(ep, "/status")
    assert proxy_recorder.connections == []


@requires_openssl
def test_a_ca_file_https_request_ignores_proxy_settings(
    proxy_recorder: _ProxyRecorder,
    tls_bundle: _CertBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With CCS_REMOTE_CA_FILE the client builds its opener per request instead of
    # taking a shared one, so the cases above never reach this path: a proxy
    # handler added only here would send the tunnel request to the proxy and
    # leave every other test green.
    monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{proxy_recorder.port}")
    srv = _start_tls_server(tls_bundle, _make_handler_class())
    try:
        _assert_a_stock_opener_uses_the_proxy(
            proxy_recorder, f"https://127.0.0.1:{srv.port}/status"
        )
        ep = CoordinatorEndpoint(
            port=srv.port,
            bearer="s3cr3t",
            host="127.0.0.1",
            scheme="https",
            ca_file=str(tls_bundle.ca_pem),
        )
        assert cc.get(ep, "/status") == {"ok": True}
        assert srv.handler_cls.seen_authorizations == ["Bearer s3cr3t"]
        assert proxy_recorder.connections == []
    finally:
        srv.shutdown()


def test_the_shared_system_trust_context_is_the_hardened_one(
    fresh_shared_openers: None,
) -> None:
    # The shared https opener must carry build_tls_context()'s context (whose
    # properties TestTlsContextFactory pins), not a bare default or unverified
    # one. The CN fallback tells them apart: both of those allow it, and the
    # shared context lasts for the whole process.
    opener = cc._get_shared_opener(system_tls=True)
    (handler,) = [
        h for h in opener.handlers if isinstance(h, urllib.request.HTTPSHandler)
    ]
    # On 3.11 a handler built without a context holds None, which has no CN
    # fallback attribute either, so check the type first.
    assert isinstance(handler._context, ssl.SSLContext)
    assert getattr(handler._context, "hostname_checks_common_name", False) is False


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_fork_during_a_first_build_does_not_hang_the_child(
    fresh_shared_openers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A forked CoherentVolume child talks to the coordinator through this module.
    # Fork while another thread holds the shared-opener lock mid-build: the child
    # inherits that lock held, with no thread left to release it.
    building, release = threading.Event(), threading.Event()
    real_build = cc._build_opener

    def held_open_build(
        context: ssl.SSLContext | None,
    ) -> urllib.request.OpenerDirector:
        building.set()
        release.wait(timeout=30)
        return real_build(context)

    monkeypatch.setattr(cc, "_build_opener", held_open_build)
    builder = threading.Thread(
        target=cc._get_shared_opener, kwargs={"system_tls": False}, name="builder"
    )
    builder.start()
    try:
        assert building.wait(timeout=10), "builder thread never started building"
        pid = os.fork()
        if pid == 0:  # child: only os._exit, never return into pytest
            exit_code = 1
            try:
                cc._build_opener = real_build
                cc._get_shared_opener(system_tls=False)
                cc._get_shared_opener(system_tls=True)
                exit_code = 0
            finally:
                os._exit(exit_code)
        deadline = time.monotonic() + 10
        while (reaped := os.waitpid(pid, os.WNOHANG))[0] == 0:
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                pytest.fail("forked child still blocked on the shared-opener lock after 10s")
            time.sleep(0.02)
        assert os.waitstatus_to_exitcode(reaped[1]) == 0
    finally:
        release.set()
        builder.join(timeout=10)


@requires_openssl
class TestSystemTrustHttpsOpener:
    """https with no CA file verifies against the system trust store. Here
    ``SSL_CERT_FILE`` points that store at a throwaway CA, so a local server can
    pass (or, pointed at an unrelated CA, fail) verification."""

    def test_system_trust_https_loads_the_trust_store_once(
        self,
        tls_bundle: _CertBundle,
        tls_setup_counts: _TlsSetupCounts,
        fresh_shared_openers: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SSL_CERT_FILE", str(tls_bundle.ca_pem))
        srv = _start_tls_server(tls_bundle, _make_handler_class())
        try:
            ep = CoordinatorEndpoint(
                port=srv.port, bearer="s3cr3t", host="127.0.0.1", scheme="https"
            )
            for _ in range(3):
                assert cc.get(ep, "/status") == {"ok": True}
            # All three were verified against the store and reached the server.
            assert srv.handler_cls.seen_authorizations == ["Bearer s3cr3t"] * 3
            assert tls_setup_counts.ca_store_loads == 1
            assert tls_setup_counts.https_handlers == 1
        finally:
            srv.shutdown()

    def test_http_and_system_trust_https_keep_separate_openers(
        self,
        tls_bundle: _CertBundle,
        fresh_shared_openers: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # One process can talk to a loopback http coordinator and a remote https
        # one. Each kind must get its own shared opener whichever is built first:
        # the plain one cannot open https, and must not gain an https handler.
        monkeypatch.setenv("SSL_CERT_FILE", str(tls_bundle.ca_pem))
        plain = _start_plain_server(_make_handler_class())
        tls = _start_tls_server(tls_bundle, _make_handler_class())
        try:
            http_ep = CoordinatorEndpoint(port=plain.port, bearer="h", host="127.0.0.1")
            https_ep = CoordinatorEndpoint(
                port=tls.port, bearer="t", host="127.0.0.1", scheme="https"
            )
            assert cc.get(http_ep, "/status") == {"ok": True}
            assert cc.get(https_ep, "/status") == {"ok": True}
            assert cc.get(http_ep, "/status") == {"ok": True}
            assert plain.handler_cls.seen_authorizations == ["Bearer h"] * 2
            assert tls.handler_cls.seen_authorizations == ["Bearer t"]
        finally:
            plain.shutdown()
            tls.shutdown()

    def test_system_trust_https_refuses_a_server_the_store_does_not_vouch_for(
        self,
        tmp_path: Path,
        tls_bundle: _CertBundle,
        fresh_shared_openers: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The shared context must still verify: with the store pointed at an
        # unrelated CA, the server's certificate is refused before any request.
        other_ca = _make_ca_pem(tmp_path / "other")
        assert other_ca is not None
        monkeypatch.setenv("SSL_CERT_FILE", str(other_ca))
        srv = _start_tls_server(tls_bundle, _make_handler_class())
        try:
            ep = CoordinatorEndpoint(
                port=srv.port, bearer="s3cr3t", host="127.0.0.1", scheme="https"
            )
            with pytest.raises(TlsVerificationFailed):
                cc.get(ep, "/status")
            assert srv.handler_cls.seen_authorizations == []
        finally:
            srv.shutdown()


@requires_openssl
class TestHttpsContextIsPerRequest:
    def test_ca_bundle_is_revalidated_on_every_request(
        self, tls_bundle: _CertBundle
    ) -> None:
        # The https path deliberately does not cache its context: a trust anchor
        # that turns group/world-writable after the first request must be refused
        # by the second, not trusted for the life of the process.
        srv = _start_tls_server(tls_bundle, _make_handler_class())
        try:
            ep = CoordinatorEndpoint(
                port=srv.port,
                bearer="s3cr3t",
                host="127.0.0.1",
                scheme="https",
                ca_file=str(tls_bundle.ca_pem),
            )
            assert cc.get(ep, "/status") == {"ok": True}
            tls_bundle.ca_pem.chmod(0o666)
            with pytest.raises(TlsConfigError, match="group/world-writable"):
                cc.get(ep, "/status")
            # Refused before connecting: the bearer rode only the first request.
            assert srv.handler_cls.seen_authorizations == ["Bearer s3cr3t"]
        finally:
            srv.shutdown()
