"""API LEARNING (cp66) -- extracted from leOS kernel_adapter_cert / api_adapter.

leOS's move, kept whole: LEARN an external API from its documentation (OpenAPI first,
no LLM anywhere in the parse), register its endpoints as callable tools, and -- the
part that makes it usable -- publish a DISCOVERABILITY CARD into the knowledge search
path, so "how do I get the weather" finds the tool the same way any other memory is
found. leOS wrote a KB article; here the card is a TEACH, which means the whole
existing contract applies for free: provenance, veto, session isolation, replay.

The call side builds the URL from the spec (path-param substitution, query params,
header auth), makes the request with stdlib urllib, parses JSON, and returns an honest
{ok, status, data | error}. Every successful call NOTES itself so usage becomes
experience (leOS records a displacement; we record through the same door).

HOSTED SAFETY, stated where it matters: arbitrary user-supplied URLs on a shared
server are an SSRF hole. The hosted zoo therefore only exposes APIs the OPERATOR
registered; per-user learning is a local-runtime feature.

SECRETS BECOME ENVIRONMENT PLACEHOLDERS (2026-09-27, owner-directed: "if the system learns how to use an api
(substituting keys for env variables or something) ... that's something that should make its way to the seed").
What is worth keeping about an API is HOW to authenticate -- which header or query parameter carries the key --
never the key. So, at LEARN time, every credential VALUE in a learned API spec (an OpenAPI securityScheme, an
`auth=` template, a key passed once to api_use, a token in a base URL), a tool reflex's fixed params and a
method row's constants is replaced by a NAMED placeholder, ${SERVICE_PARAM} (placeholder_name: deterministic from
service + parameter: toyair + X-Api-Key -> ${TOYAIR_API_KEY}). The value is read from os.environ only at CALL time
(resolve_placeholders); a variable that is not set FAILS LOUDLY naming it -- the call is never made without the
auth it needs. Learned state written before this rule that still holds a raw key is migrated on load (the toolbox
rehydrate here, the tool-reflex rebuild in p19, MeaningIndex.from_state). The detectors are the learning guard's
(holographic_learnguard: sensitive_reason, _credential_shaped, the _STRONG shapes and the credential-name rule of
redact_args) -- one definition of "a secret" for the whole engine.
KEPT NEGATIVE, stated: a credential under an innocuous NAME with an innocuous SHAPE (a 10-character lower-case
token in a param called "q") is not detected -- there is nothing to detect it by. The guard is the floor, and the
distiller's final secret scan (tools/distill_release.py) is the second net.
"""
import json
import os
import re
import urllib.request
import urllib.parse
import urllib.error

import numpy as np


# ------------------------------------------------------------------------------------------ secret placeholders
# ${NAME}: upper-case letters, digits, underscore -- the POSIX environment-variable shape, so a placeholder can be
# exported in any shell as-is.
PLACEHOLDER_RX = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
# A NAME that says "this value is a credential". The redact_args rule (key/token/secret/pass/pwd/seed as a whole
# _-separated word) plus the auth vocabulary real APIs use: Authorization headers, OpenWeather's `appid`, `apikey`,
# `X-Api-Key`, `access_key`, `client_secret`, request signatures.
_CRED_NAME = re.compile(
    r"(?i)^(?:x-)?(?:authorization|proxy-authorization|appid|app[_-]?id|app[_-]?key|api[_-]?key|apikey|"
    r"access[_-]?key|secret[_-]?key|client[_-]?secret|auth[_-]?token|access[_-]?token|refresh[_-]?token|"
    r"bearer|token|key|secret|password|passwd|pwd|signature|sig)$")
# ...and a name that merely CONTAINS a credential word as a whole _/- separated part ("stripe_secret", "X-Auth-Token",
# "primary_key"): a credential only when the value is also credential-shaped (6+ characters, no whitespace), so a
# database's "primary_key": "id" stays a parameter. "seed" is NOT in this list (redact_args has it): an API's
# "random_seed": "42" is a parameter, and a false placeholder here makes a call fail for a variable nobody needs.
_CRED_PART = re.compile(r"(?i)(?:^|[_-])(?:key|token|secret|pass|pwd|password|auth)(?:$|[_-])")
# Hosts whose spec must NOT travel into a shared memory: a user's own machine or LAN service is theirs.
_PRIVATE_HOST = re.compile(r"^(?:localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
                           r"172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|\[?::1\]?|.*\.local|.*\.internal|.*\.lan)$", re.I)


class MissingCredential(KeyError):
    """A ${VAR} placeholder whose environment variable is not set. Raised (or reported) BEFORE any request is
    made -- a learned call never goes out silently without the auth it was learned with."""

    def __init__(self, names):
        self.names = sorted(set(names))
        super().__init__("set the environment variable%s %s (a learned credential placeholder)"
                         % ("s" if len(self.names) > 1 else "", ", ".join(self.names)))


def _slug(text):
    """UPPER_SNAKE of any text: 'toy weather' -> 'TOY_WEATHER', 'X-Api-Key' -> 'X_API_KEY'."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", str(text))).strip("_").upper()


def placeholder_name(service, param):
    """The environment variable a credential lives in: SERVICE_PARAM, deterministic from the two names.
    The 'x-' header prefix is dropped (X-Api-Key -> API_KEY); an Authorization header or a bearer token is TOKEN;
    a param that already starts with the service name is not prefixed twice.
        placeholder_name('openweather', 'appid')     -> 'OPENWEATHER_APPID'
        placeholder_name('toyair', 'X-Api-Key')      -> 'TOYAIR_API_KEY'
        placeholder_name('github', 'Authorization')  -> 'GITHUB_TOKEN'"""
    svc = _slug(service) or "SERVICE"
    p = re.sub(r"(?i)^x-", "", str(param or ""))
    p = "TOKEN" if re.fullmatch(r"(?i)(proxy-)?authorization|bearer", p) else _slug(p) or "SECRET"
    if svc and (p == svc or p.startswith(svc + "_")):
        return p if p[:1].isalpha() else "V_" + p
    name = "%s_%s" % (svc, p)
    return name if name[:1].isalpha() else "V_" + name


def _guard():
    # the learning guard is the one definition of "a secret"; imported lazily (it is a big module and this one
    # loads on the api_toolbox() path only)
    from holographic.agents_and_reasoning import holographic_learnguard as LG
    return LG


def is_credential(name, value):
    """Is `value` (stored under `name`) a credential? A credential NAME with a non-empty, non-placeholder value;
    or a value the learning guard calls sensitive on its own (an sk- key, a JWT, a GitHub token ...) whatever it is
    called. Non-strings are never credentials (a port, a count)."""
    if not isinstance(value, str) or not value.strip() or PLACEHOLDER_RX.search(value):
        return False
    LG = _guard()
    v = value.strip()
    if LG.sensitive_reason(v):
        return True
    if LG._PLACEHOLDER.match(v):
        # "documentation" values (required / string / <your key> / os.environ...) are not secrets: the guard's own
        # placeholder vocabulary decides, so a spec's example text is left as it is
        return False
    nm = str(name or "")
    if _CRED_NAME.search(nm):
        return True
    return bool(_CRED_PART.search(nm)) and LG._credential_shaped(v)


def _sub_value(name, value, service, found):
    """The placeholder for one credential value. 'Bearer <tok>' keeps its scheme word: 'Bearer ${SVC_TOKEN}'."""
    var = placeholder_name(service, name)
    found.append(var)
    m = re.match(r"(?i)^(bearer|token|basic)\s+\S+$", value.strip())
    if m:
        return "%s ${%s}" % (m.group(1), var)
    return "${%s}" % var


def placeholderize_url(url, service, found=None):
    """A URL with every credential QUERY parameter (?appid=..., &api_key=...) and any user:password@ turned into
    placeholders. The path is left alone (a token in a path segment is caught by placeholderize_text's shapes)."""
    found = [] if found is None else found
    s = str(url)
    try:
        u = urllib.parse.urlsplit(s)
    except ValueError:
        return s
    if not u.scheme or not u.netloc:
        return _text_rules(s, service or "SECRET", found)
    netloc = u.netloc
    if "@" in netloc:                                   # https://user:pass@host -> the password is a credential
        cred, host = netloc.rsplit("@", 1)
        if ":" in cred:
            user = cred.split(":", 1)[0]
            netloc = "%s:${%s}@%s" % (user, placeholder_name(service, "password"), host)
            found.append(placeholder_name(service, "password"))
    q = urllib.parse.parse_qsl(u.query, keep_blank_values=True)
    q2 = [(k, _sub_value(k, v, service, found) if is_credential(k, v) else v) for k, v in q]
    # urlencode would escape ${...}: rebuild by hand, quoting only non-placeholder values
    query = "&".join("%s=%s" % (urllib.parse.quote(k, safe=""), v if PLACEHOLDER_RX.fullmatch(v) else
                                urllib.parse.quote(v, safe="")) for k, v in q2)
    out = urllib.parse.urlunsplit((u.scheme, netloc, u.path, query if q2 else u.query, u.fragment))
    return _text_rules(out, service or "SECRET", found)   # a token-shaped PATH segment (strong shapes)


def placeholderize_text(text, service=None, found=None):
    """Free text (an SOP, a taught answer, a URL) with every credential span replaced by a named placeholder:
      * `name = value` / `"name": "value"` assignments the guard recognises   -> ${SERVICE_NAME}
      * URL query parameters with a credential name                           -> ${HOST_NAME} (placeholderize_url)
      * a strong shape anywhere (sk-..., ghp_..., a JWT, AKIA...)              -> ${SERVICE_SECRET}[_2, _3 ...]
    `service` names the variable's prefix; without one, a URL's host names it (api.openweathermap.org ->
    OPENWEATHERMAP), else 'SECRET'. What is NOT a credential API value -- a wallet seed phrase, a PEM private key,
    a 64-byte keypair array -- is left for the guard to REFUSE: those never become placeholders (a seed phrase is
    not something a learned call sends, and the row carrying one must not ship at all)."""
    found = [] if found is None else found
    s = str(text)
    svc = service
    if not svc:
        mh = re.search(r"https?://(?:[^/@\s]*@)?([A-Za-z0-9.-]+)", s)
        if mh:
            parts = [p for p in mh.group(1).lower().split(".") if p not in ("www", "api", "com", "org", "net",
                                                                            "io", "co", "dev", "app")]
            svc = parts[0] if parts else None
    svc = svc or "SECRET"
    # URLs first: their query parameters have names
    s = re.sub(r"https?://[^\s\"'<>)]+", lambda m: placeholderize_url(m.group(0), svc, found)
               if "?" in m.group(0) or "@" in m.group(0) else m.group(0), s)
    return _text_rules(s, svc, found)


def _text_rules(s, svc, found):
    """The non-URL half of placeholderize_text: named assignments, then unnamed strong token shapes."""
    LG = _guard()
    # assignments: the VALUE group of each hit (right to left, de-overlapped -- redact()'s discipline)
    spans = []
    for m in LG._ASSIGN.finditer(s):
        if not LG._PLACEHOLDER.match(m.group(2)):
            spans.append((m.start(2), m.end(2), m.group(1)))
    for m in re.finditer(r"(?i)\b(authorization|x-api-key|api-key|apikey|appid)[\"']?\s*[:=]\s*[\"']?"
                         r"((?:bearer|token|basic)\s+)?([^\s\"',;}]{6,})", s):
        if not LG._PLACEHOLDER.match(m.group(3)) and not PLACEHOLDER_RX.search(m.group(3)):
            spans.append((m.start(3), m.end(3), m.group(1)))
    last = None
    for a, b, nm in sorted(set(spans), key=lambda x: (-x[0], x[1])):
        if last is not None and b > last:
            continue
        var = placeholder_name(svc, nm)
        found.append(var)
        s = s[:a] + "${%s}" % var + s[b:]
        last = a
    # strong API-token shapes with no name around them (PEM keys and keypair arrays are refused, not replaced)
    n = [0]

    def _strong(m):
        n[0] += 1
        var = placeholder_name(svc, "secret" if n[0] == 1 else "secret_%d" % n[0])
        found.append(var)
        return "${%s}" % var
    for label, rx in LG._STRONG:
        if "PEM" in label or "keypair" in label:
            continue
        s = rx.sub(_strong, s)
    return s


def placeholderize(obj, service, found=None, _name=None):
    """A COPY of a learned structure (spec, params, headers, method args) with every credential value replaced by
    ${SERVICE_NAME}. Dict keys give names; strings are also scanned as text. Returns the copy; `found` (a list)
    collects the variable names used, so the caller can tell the person which ones to export."""
    found = [] if found is None else found
    if isinstance(obj, dict):
        return {k: placeholderize(v, service, found, _name=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [placeholderize(v, service, found, _name=_name) for v in obj]
    if isinstance(obj, tuple):
        return tuple(placeholderize(v, service, found, _name=_name) for v in obj)
    if isinstance(obj, str):
        if is_credential(_name, obj):
            return _sub_value(_name, obj, service, found)
        if "://" in obj:
            return placeholderize_url(obj, service, found)
        return placeholderize_text(obj, service, found) if len(obj) >= 16 else obj
    return obj


def placeholders_in(obj):
    """Every ${VAR} name in a structure (sorted, unique)."""
    return sorted(set(PLACEHOLDER_RX.findall(json.dumps(obj, default=str) if not isinstance(obj, str) else obj)))


def resolve_placeholders(obj, env=None):
    """A copy of `obj` with every ${VAR} read from the environment (os.environ unless `env` is given) -- the ONLY
    place a learned credential becomes a value, at call time. Raises MissingCredential naming every unset variable
    (all of them at once, so one error message is the whole to-do list)."""
    env = os.environ if env is None else env
    missing = [v for v in placeholders_in(obj) if not env.get(v)]
    if missing:
        raise MissingCredential(missing)

    def _r(x):
        if isinstance(x, dict):
            return {k: _r(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_r(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_r(v) for v in x)
        if isinstance(x, str):
            return PLACEHOLDER_RX.sub(lambda m: env[m.group(1)], x)
        return x
    return _r(obj)


def private_host(url):
    """True for a URL on this machine or a private network (localhost, 127/8, 10/8, 192.168/16, 172.16/12,
    *.local, *.internal): a spec for such a host is one user's own service and never travels into shared memory."""
    try:
        h = urllib.parse.urlsplit(str(url)).hostname or ""
    except ValueError:
        return False
    return bool(h) and bool(_PRIVATE_HOST.match(h))


def _auth_from_spec(spec, service):
    """The auth TEMPLATE an OpenAPI spec declares (components.securitySchemes, or swagger 2 securityDefinitions):
    apiKey in header / query -> that name carries ${SERVICE_NAME}; http bearer / oauth2 -> Authorization: Bearer
    ${SERVICE_TOKEN}; http basic -> Authorization: Basic ${SERVICE_BASIC}. Learned from the documentation with no
    key in sight -- how to authenticate is part of how to use the API."""
    schemes = ((spec.get("components") or {}).get("securitySchemes") or spec.get("securityDefinitions") or {})
    auth = {"headers": {}, "query": {}}
    for _sname, sc in sorted(schemes.items()):
        if not isinstance(sc, dict):
            continue
        t = str(sc.get("type", "")).lower()
        if t == "apikey" and sc.get("name"):
            where = "query" if str(sc.get("in", "header")).lower() == "query" else "headers"
            auth[where][str(sc["name"])] = "${%s}" % placeholder_name(service, sc["name"])
        elif (t == "http" and str(sc.get("scheme", "")).lower() == "bearer") or t in ("oauth2", "openidconnect"):
            auth["headers"]["Authorization"] = "Bearer ${%s}" % placeholder_name(service, "Authorization")
        elif t == "http" and str(sc.get("scheme", "")).lower() == "basic":
            auth["headers"]["Authorization"] = "Basic ${%s}" % placeholder_name(service, "basic")
        elif t == "basic":
            auth["headers"]["Authorization"] = "Basic ${%s}" % placeholder_name(service, "basic")
    return auth if (auth["headers"] or auth["query"]) else None


class ApiToolbox:
    """Learn APIs from specs; call them by name; find them by task."""

    # NOT "__api_spec__": the cp38 control-token guard refuses standalone __token__
    # words on either side of a remember -- correctly, that is the attack class the
    # harsh battery taught it to refuse -- and it ate the first version of this
    # record. The registry rides as plain words instead; the guard stays untouched.
    SPEC_PREFIX = "api spec record: "

    def __init__(self, mind=None):
        self.mind = mind
        self.services = {}
        self._rehydrated = False
        self.migrated = 0           # spec records rewritten from a raw credential to placeholders on load

    def _rehydrate(self):
        """RESTART SURVIVAL (cp67, found by the battery): the discoverability cards
        persisted but the service registry died with the process -- after a restart
        every card pointed at an empty toolbox. Each learned spec is therefore
        TAUGHT as a record ("__api_spec__ <service>" -> spec json) and the toolbox
        rebuilds from those records lazily. Replay-safe by construction, and
        VETO-ABLE: vetoing a spec record un-registers the service, the same lever
        as everything else.
        MIGRATION ON LOAD (2026-09-27): a record written before the placeholder rule that still holds a raw
        credential (a key in the base URL, an auth header value) is rewritten to placeholders HERE -- the taught row
        is replaced in place (the raw text is gone from the next save) and re-taught, so the ladder's exact store
        serves the migrated record too. self.migrated counts them."""
        if self._rehydrated or self.mind is None:
            return
        self._rehydrated = True
        lad = self.mind.zoo.get("ladder")
        log = getattr(lad, "taught_log", []) or []
        redo = []
        for i, t in enumerate(log):
            q = str(t[0])
            if q.startswith(self.SPEC_PREFIX) and (len(t) < 4 or
                                                   t[3] != "model-cached"):
                name = q[len(self.SPEC_PREFIX):].strip()
                try:
                    spec = json.loads(str(t[1]))
                except Exception:
                    continue
                clean = self._clean_spec(name, spec)
                if clean != spec:
                    log[i] = [t[0], json.dumps(clean)] + list(t[2:])
                    redo.append((q, clean))
                self.services[name] = clean
        for q, clean in redo:
            self.migrated += 1
            try:
                self.mind.teach(q, json.dumps(clean))
            except Exception:
                pass

    @staticmethod
    def _clean_spec(service, svc):
        """A stored service record with every credential value turned into ${SERVICE_PARAM} placeholders: the
        base URL (?apikey=..., user:pass@), the auth template, and any endpoint text. Idempotent."""
        out = dict(svc)
        out["base"] = placeholderize_url(out.get("base", ""), service) if out.get("base") else out.get("base", "")
        if out.get("auth"):
            out["auth"] = {"headers": placeholderize(dict((out["auth"] or {}).get("headers") or {}), service),
                           "query": placeholderize(dict((out["auth"] or {}).get("query") or {}), service)}
        return out

    # -- learning ---------------------------------------------------------
    def learn(self, spec, name=None, base_url=None, teach=True, auth=None):
        """Ingest an OpenAPI spec (dict, JSON text, or a URL to fetch it from).
        Registers every operation; returns {service, endpoints}. With a mind
        attached and teach=True, each endpoint gets a discoverability card taught
        into memory -- contextual access is then ordinary recall.
        AUTH (2026-09-27): how the API authenticates is learned as a TEMPLATE of placeholders -- from the spec's
        securitySchemes, and/or from `auth` = {"headers": {...}, "query": {...}} (raw values given here are turned
        into ${SERVICE_PARAM} placeholders before anything is stored). The result names the environment variables
        (`env`) the calls will read."""
        if isinstance(spec, str) and spec.strip().startswith(("http://",
                                                              "https://")):
            with urllib.request.urlopen(spec, timeout=10) as r:
                spec = r.read().decode()
        if isinstance(spec, str):
            spec = json.loads(spec)
        title = name or re.sub(r"\W+", "_",
                               spec.get("info", {}).get("title", "api")).lower()
        base = base_url or (spec.get("servers") or [{}])[0].get("url", "")
        eps = {}
        for path, methods in (spec.get("paths") or {}).items():
            for method, op in methods.items():
                if method.upper() not in ("GET", "POST", "PUT", "DELETE"):
                    continue
                eid = op.get("operationId") or re.sub(
                    r"\W+", "_", "%s %s" % (method, path)).strip("_").lower()
                params = [{"name": p["name"], "in": p.get("in", "query"),
                           "required": bool(p.get("required"))}
                          for p in op.get("parameters", [])]
                eps[eid] = {"method": method.upper(), "path": path,
                            "params": params,
                            "description": op.get("summary") or
                            op.get("description") or eid}
        rec = {"base": base, "endpoints": eps}
        tmpl = _auth_from_spec(spec, title)
        if auth:
            tmpl = tmpl or {"headers": {}, "query": {}}
            for where in ("headers", "query"):
                for k, v in ((auth or {}).get(where) or {}).items():
                    # a raw value given here is the key itself: it becomes a placeholder, never stored
                    tmpl[where][str(k)] = v if PLACEHOLDER_RX.search(str(v)) else \
                        _sub_value(str(k), str(v), title, [])
        if tmpl:
            rec["auth"] = tmpl
        rec = self._clean_spec(title, rec)
        self.services[title] = rec
        if self.mind is not None:
            self.mind.teach(self.SPEC_PREFIX + title,
                            json.dumps(self.services[title]))
        if teach and self.mind is not None:
            for eid, ep in eps.items():
                self.mind.teach(
                    "how do i %s" % ep["description"].lower().rstrip("."),
                    "use the learned api tool %s.%s -- %s %s%s with params %s; "
                    "call it via api_use(%r, %r, params={...})"
                    % (title, eid, ep["method"], rec["base"], ep["path"],
                       [p["name"] for p in ep["params"]], title, eid))
        return {"service": title, "base": rec["base"], "endpoints": sorted(eps),
                "env": placeholders_in(rec)}

    def _learn_auth(self, service, params, headers):
        """A call that SUCCEEDED with a credential the caller passed by hand teaches HOW this service
        authenticates: the header / query name is remembered with a ${SERVICE_PARAM} placeholder (the value is
        never kept) and the spec record is re-taught. -> the new {where: {name: placeholder}} entries (or {})."""
        svc = self.services.get(service)
        if svc is None:
            return {}
        auth = svc.setdefault("auth", {"headers": {}, "query": {}})
        auth.setdefault("headers", {})
        auth.setdefault("query", {})
        declared = {p["name"] for ep in svc["endpoints"].values() for p in ep["params"]}
        new = {}
        for where, src in (("headers", headers or {}), ("query", params or {})):
            for k, v in src.items():
                if where == "query" and k in declared and not _CRED_NAME.search(str(k)):
                    continue                            # an endpoint parameter, not auth
                if is_credential(k, v) and not PLACEHOLDER_RX.search(str(auth[where].get(k, ""))):
                    auth[where][str(k)] = _sub_value(str(k), str(v), service, [])
                    new.setdefault(where, {})[str(k)] = auth[where][str(k)]
        if new and self.mind is not None:
            self.mind.teach(self.SPEC_PREFIX + service, json.dumps(svc))
        return new

    # -- calling ----------------------------------------------------------
    def call(self, service, endpoint, params=None, headers=None, timeout=10, env=None):
        """Call a learned endpoint. Stored ${VAR} placeholders (in the base URL, the auth template, a tool reflex's
        fixed params) are read from os.environ (or `env`) NOW; an unset variable returns ok=False naming it
        (`missing_env`) and NO request is made. A credential passed by hand in params / headers is used for this
        call and, if the call succeeds, learned as a placeholder (`learned_auth`) -- never as a value."""
        self._rehydrate()
        svc = self.services.get(service)
        if not svc or endpoint not in svc["endpoints"]:
            return {"ok": False, "error": "unknown %s.%s -- learned services: %s"
                    % (service, endpoint, sorted(self.services))}
        ep = svc["endpoints"][endpoint]
        given_params, given_headers = dict(params or {}), dict(headers or {})
        auth = svc.get("auth") or {}
        # the auth template fills what the caller did not pass explicitly (an explicit value wins for this call)
        params = dict((auth.get("query") or {}), **given_params)
        hdrs = dict((auth.get("headers") or {}), **given_headers)
        try:
            base, params, hdrs = resolve_placeholders([svc["base"], params, hdrs], env=env)
        except MissingCredential as e:
            return {"ok": False, "missing_env": e.names,
                    "error": "not called: %s.%s needs %s -- export %s (learned credentials are read from the "
                             "environment at call time, never stored)" % (service, endpoint, ", ".join(e.names),
                                                                           " and ".join(e.names))}
        missing = [p["name"] for p in ep["params"]
                   if p.get("required") and p["name"] not in params]
        if missing:
            return {"ok": False, "error": "missing required param(s) %s for %s.%s"
                    % (missing, service, endpoint)}
        path = ep["path"]
        for p in ep["params"]:
            if p["in"] == "path" and p["name"] in params:
                path = path.replace("{%s}" % p["name"],
                                    urllib.parse.quote(str(params.pop(p["name"]))))
        url = base.rstrip("/") + path
        q = {k: v for k, v in params.items()}
        if ep["method"] == "GET" and q:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(q)
            body = None
        else:
            body = json.dumps(q).encode() if q else None
        req = urllib.request.Request(url, data=body, method=ep["method"],
                                     headers={"Content-Type": "application/json",
                                              **hdrs})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
                try:
                    data = json.loads(raw)
                except Exception:
                    data = raw[:2000]
                out = {"ok": True, "status": r.status, "data": data}
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode()
                body = json.loads(raw)
            except Exception:
                body = raw[:500]
            return {"ok": False, "status": int(e.code), "data": body,
                    "error": "HTTP %d" % e.code}
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__,
                                                      str(e)[:200])}
        learned = self._learn_auth(service, given_params, given_headers)
        if learned:
            out["learned_auth"] = learned
            out["env"] = placeholders_in(learned)
        if self.mind is not None:
            try:
                self.mind.drift_sentinel().note(
                    self.mind.semantic_key("%s.%s" % (service, endpoint))
                    ["vec"][:64],
                    self.mind.semantic_key(str(out["data"])[:200])["vec"][:64])
            except Exception:
                pass
        return out

    # -- contextual discovery --------------------------------------------
    def find(self, task, k=3):
        """Rank every learned endpoint against a task description -- semantic
        match plus the grounding doctrine (shared substantive token required)."""
        self._rehydrate()
        rows = []
        tt = {w for w in str(task).lower().split() if len(w) >= 4}
        for sname, svc in self.services.items():
            for eid, ep in svc["endpoints"].items():
                text = "%s %s %s" % (sname, eid, ep["description"])
                et = {w for w in text.lower().split() if len(w) >= 4}
                overlap = len(tt & et)
                if overlap:
                    rows.append({"tool": "%s.%s" % (sname, eid),
                                 "description": ep["description"][:100],
                                 "score": overlap})
        rows.sort(key=lambda r: -r["score"])
        return rows[:k]


def _selftest():
    """A REAL http loop, closed locally: serve a toy API in-process, learn it from
    its OpenAPI spec, discover the endpoint from a task phrase, call it over actual
    HTTP, and verify the discoverability card serves from memory."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/secure"):
                # the auth'd endpoint: 401 unless the FAKE test key arrives in the header
                ok_ = self.headers.get("X-Api-Key") == "Hunter2-FAKE-9c1d-apikey"
                body = json.dumps({"ok": ok_}).encode()
                self.send_response(200 if ok_ else 401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/temp/"):
                city = self.path.split("/temp/")[1]
                body = json.dumps({"city": urllib.parse.unquote(city),
                                   "temp_c": 21.5}).encode()
            else:
                body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    spec = {"info": {"title": "toy weather"},
            "servers": [{"url": "http://127.0.0.1:%d" % port}],
            "paths": {"/temp/{city}": {"get": {
                "operationId": "get_temperature",
                "summary": "get the current temperature for a city",
                "parameters": [{"name": "city", "in": "path",
                                "required": True}]}}}}
    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    box = ApiToolbox(mind=m)
    rep = box.learn(spec)
    assert rep["endpoints"] == ["get_temperature"], rep
    hit = box.find("what is the temperature in a city")
    assert hit and hit[0]["tool"] == "toy_weather.get_temperature", hit
    r = box.call("toy_weather", "get_temperature", params={"city": "lisbon"})
    assert r["ok"] and r["data"]["temp_c"] == 21.5 and r["data"]["city"] == "lisbon"
    card = m.ask("how do i get the current temperature for a city")
    assert "toy_weather.get_temperature" in str(card.get("answer")), \
        "the discoverability card serves from ordinary memory (the leOS KB move)"
    bad = box.call("toy_weather", "nope")
    assert not bad["ok"] and "unknown" in bad["error"]
    miss = box.call("toy_weather", "get_temperature")
    assert not miss["ok"] and "missing required" in miss["error"]
    import tempfile as _tf, shutil as _sh
    _d = _tf.mkdtemp(prefix="apiln_")
    m.learning_save(_d)
    m2 = lecore.UnifiedMind()
    m2.zoo_attach(lambda p: "")
    m2.learning_load(_d)
    box2 = ApiToolbox(mind=m2)
    r2 = box2.call("toy_weather", "get_temperature", params={"city": "faro"})
    assert r2["ok"] and r2["data"]["city"] == "faro", \
        "the toolbox survives a restart by rehydrating from taught spec records"
    # SECRETS -> PLACEHOLDERS (2026-09-27): the key is passed ONCE by hand; the successful call teaches the header
    # name with ${TOY_SECURE_API_KEY}, never the value; the next call reads the environment, and an unset variable
    # fails loudly WITHOUT calling
    fake = "Hunter2-FAKE-9c1d-apikey"
    box.learn({"info": {"title": "toy secure"}, "servers": [{"url": "http://127.0.0.1:%d" % port}],
               "paths": {"/secure": {"get": {"operationId": "ping", "summary": "ping the secure endpoint"}}}})
    r3 = box.call("toy_secure", "ping", headers={"X-Api-Key": fake})
    assert r3["ok"] and r3["env"] == ["TOY_SECURE_API_KEY"], r3
    assert fake not in json.dumps(box.services) and fake not in json.dumps(
        [list(t) for t in m.zoo["ladder"].taught_log]), "the raw key must never be stored"
    miss2 = box.call("toy_secure", "ping", env={})
    assert not miss2["ok"] and miss2["missing_env"] == ["TOY_SECURE_API_KEY"], miss2
    r4 = box.call("toy_secure", "ping", env={"TOY_SECURE_API_KEY": fake})
    assert r4["ok"], r4
    assert placeholder_name("openweather", "appid") == "OPENWEATHER_APPID"
    assert placeholderize_url("https://api.x.io/v1?q=lisbon&appid=Hunter2FAKE9c1d77", "x") == \
        "https://api.x.io/v1?q=lisbon&appid=${X_APPID}"
    _sh.rmtree(_d, ignore_errors=True)
    srv.shutdown()
    return ("OK: learned an API from its spec with no LLM, found the endpoint from "
            "a task phrase, called it over real HTTP (127.0.0.1:%d, temp 21.5), and "
            "the discoverability card serves from memory with full provenance" % port)


if __name__ == "__main__":
    print(_selftest())
