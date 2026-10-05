"""g1: deterministic (seeded, no network) HARD synthetic retrieval corpus generator.

240 concepts = 30 procedure topics x 8 sibling details.  Siblings (same topic, different detail) are near-duplicate docs that
differ in ONE key detail (environment / database / cloud / language / OS / payment provider / CI / broker / platform /
file format / region).  Each detail has 3 surface forms (synonyms/abbreviations); each concept has TWO multi-sentence docs
that together use all three forms, and queries use any form (so some queries match a doc only semantically, not lexically).
Plus 300 unrelated off-topic docs and 200 off-topic (OOD) queries.
Queries (8 per concept): 3 clean paraphrases, 2 typo variants, 2 terse keyword queries, 1 chatty query with irrelevant preamble.
Each query also gets two deterministic REWRITES computed from the doc vocabulary only: spell-corrected, and spell-corrected
+ keyword-extracted (drop OOV / very common words).

Usage: python py/g1_corpus.py <seed> <out.json>      (A = seed 1 -> results/g1_corpus_A.json, B = seed 2 -> g1_corpus_B.json)
Imports only json/random/re/sys/collections.
"""
import collections, json, random, re, sys

POOLS = {
    "env": [("staging", "pre-production", "pre-prod"), ("production", "live", "prod"), ("development", "dev", "local dev"),
            ("QA", "quality assurance", "test"), ("sandbox", "playground", "demo"), ("disaster recovery", "DR site", "failover site"),
            ("canary", "early-access", "beta ring"), ("load-testing", "perf", "benchmark")],
    "db": [("PostgreSQL", "Postgres", "pg"), ("MySQL", "MariaDB-style", "mysql"), ("MongoDB", "Mongo", "document store"),
           ("Redis", "Redis cache", "key-value cache"), ("SQLite", "embedded SQLite", "file-based database"),
           ("Cassandra", "Cassandra cluster", "wide-column store"), ("Elasticsearch", "Elastic", "search index"),
           ("DynamoDB", "Dynamo", "AWS key-value table")],
    "cloud": [("AWS", "Amazon Web Services", "EC2"), ("Azure", "Microsoft cloud", "Azure subscription"),
              ("GCP", "Google Cloud", "Google compute"), ("DigitalOcean", "DO droplets", "Droplet"),
              ("Hetzner", "Hetzner Cloud", "German VPS host"), ("on-premises", "bare metal", "in-house datacenter"),
              ("Oracle Cloud", "OCI", "Oracle's cloud"), ("Alibaba Cloud", "Aliyun", "Alibaba ECS")],
    "lang": [("Python", "py", "Python 3"), ("Rust", "Rustlang", "cargo"), ("Go", "Golang", "Go module"), ("Java", "JVM", "Java Spring"),
             ("TypeScript", "TS", "typed Node"), ("C#", ".NET", "dotnet"), ("Ruby", "Rails", "Ruby on Rails"), ("PHP", "Laravel", "PHP")],
    "os": [("Ubuntu", "Debian-based", "apt-based"), ("RHEL", "Red Hat", "yum-based"), ("Windows Server", "Windows", "Microsoft server"),
           ("macOS", "Mac", "Apple laptop"), ("Alpine", "Alpine Linux", "musl-based"), ("Arch", "Arch Linux", "rolling-release"),
           ("FreeBSD", "BSD", "ZFS-based Unix"), ("Fedora", "Fedora Linux", "dnf-based")],
    "pay": [("Stripe", "Stripe payments", "card processor Stripe"), ("PayPal", "PayPal checkout", "PayPal wallet"),
            ("Adyen", "Adyen gateway", "Dutch payment platform"), ("Square", "Square POS", "Square terminal"),
            ("Klarna", "buy-now-pay-later", "Klarna instalments"), ("Braintree", "Braintree gateway", "Braintree vault"),
            ("Apple Pay", "Apple wallet", "tap-to-pay on iPhone"), ("bank transfer", "SEPA", "wire")],
    "ci": [("GitHub Actions", "GH workflows", "Actions"), ("GitLab CI", "GitLab runners", "gitlab-ci"),
           ("Jenkins", "Jenkins pipeline", "Jenkinsfile"), ("CircleCI", "Circle", "CircleCI orbs"),
           ("Travis", "Travis CI", "travis.yml"), ("TeamCity", "JetBrains CI", "TeamCity agents"),
           ("Azure DevOps", "Azure Pipelines", "ADO"), ("Buildkite", "Buildkite agents", "Buildkite pipeline")],
    "msg": [("Kafka", "Kafka topic", "event log"), ("RabbitMQ", "Rabbit", "AMQP broker"), ("SQS", "AWS queue", "Simple Queue Service"),
            ("NATS", "NATS server", "NATS JetStream"), ("Pulsar", "Apache Pulsar", "Pulsar cluster"),
            ("Redis Streams", "Redis stream", "Redis consumer groups"), ("ActiveMQ", "Active MQ", "JMS broker"),
            ("Google Pub/Sub", "GCP Pub/Sub", "Pub/Sub")],
    "plat": [("iOS", "iPhone", "Apple mobile"), ("Android", "Android phone", "Google mobile"), ("React Native", "RN", "cross-platform JS mobile"),
             ("Flutter", "Dart mobile", "Flutter"), ("Windows desktop", "Win32", "Windows client"), ("macOS desktop", "Mac client", "Cocoa"),
             ("Linux desktop", "GNOME", "Linux client"), ("web PWA", "progressive web app", "installable web")],
    "fmt": [("CSV", "comma-separated", "csv"), ("JSON", "JSON document", "JSON payload"), ("XML", "XML feed", "XML document"),
            ("Parquet", "columnar Parquet", "parquet"), ("YAML", "YAML config", "yaml"), ("PDF", "PDF document", "scanned PDF"),
            ("Excel", "XLSX", "spreadsheet"), ("Avro", "Avro record", "Avro")],
    "reg": [("EU", "Europe", "Frankfurt"), ("US-East", "Virginia", "N. Virginia"), ("US-West", "Oregon", "west coast"),
            ("APAC", "Singapore", "Asia-Pacific"), ("UK", "London", "British"), ("Brazil", "Sao Paulo", "South American"),
            ("India", "Mumbai", "Indian"), ("Canada", "Montreal", "Canadian")],
}

# (pool, [4 paraphrases of the procedure with {D}])
TOPICS = [
    ("env", ["rotate the API credentials in the {D} environment", "swap out the access keys used in {D}", "refresh service secrets for {D}", "renew the tokens our apps use in {D}"]),
    ("env", ["freeze deployments to {D}", "put a change freeze on {D}", "block new releases from reaching {D}", "halt shipping code into {D}"]),
    ("env", ["scale down the idle servers in {D} overnight", "shut off unused compute in {D} after hours", "cut {D} capacity at night", "power down spare machines in {D} outside work hours"]),
    ("db", ["restore a backup of the {D} database", "bring back a {D} database from a snapshot", "recover {D} data from a saved dump", "get a {D} instance back from its backup"]),
    ("db", ["tune slow queries on {D}", "speed up sluggish lookups in {D}", "optimize performance problems on {D}", "fix queries that crawl on {D}"]),
    ("db", ["upgrade {D} to the next major version", "move our {D} to a newer release", "do a major version bump of {D}", "migrate {D} onto the latest major"]),
    ("db", ["set up replication for {D}", "configure a standby copy of {D}", "make a read replica of {D}", "mirror our {D} data to a second node"]),
    ("cloud", ["deploy the application to {D}", "ship our service onto {D}", "release the app on {D}", "get our software running on {D}"]),
    ("cloud", ["reduce the monthly bill on {D}", "cut hosting costs in {D}", "trim cloud spending on {D}", "lower what we pay {D} each month"]),
    ("cloud", ["set up a private network and firewall rules on {D}", "lock down network access in {D}", "configure isolated subnets and security rules on {D}", "restrict inbound traffic in our {D} setup"]),
    ("lang", ["configure structured logging for the {D} service", "add JSON logs to our {D} app", "set up log output in the {D} microservice", "make a {D} service emit machine-readable logs"]),
    ("lang", ["profile memory leaks in a {D} program", "find what is eating RAM in a {D} app", "track down heap growth in {D} code", "diagnose runaway memory use in our {D} service"]),
    ("lang", ["write unit tests and mocks for the {D} codebase", "add automated tests to a {D} project", "cover our {D} module with test cases", "build a test suite with fakes for the {D} code"]),
    ("lang", ["package a {D} app into a Docker image", "containerize the {D} application", "build a container for our {D} service", "create an image of our {D} program"]),
    ("os", ["apply security patches on {D} servers", "install urgent updates on {D} machines", "patch the vulnerabilities on {D} hosts", "roll out critical fixes to {D} boxes"]),
    ("os", ["set up automatic disk encryption on {D}", "turn on full-disk encryption for {D}", "encrypt the drives of {D} systems", "enable encrypted storage on our {D} computers"]),
    ("os", ["troubleshoot a machine that will not boot on {D}", "fix a {D} box that fails to start up", "diagnose boot failures on {D}", "recover a {D} computer stuck during startup"]),
    ("pay", ["issue a refund through {D}", "send money back to a customer via {D}", "process a customer reimbursement on {D}", "return a payment using {D}"]),
    ("pay", ["reconcile monthly payouts from {D}", "match {D} settlement reports against the ledger", "check {D} deposits against our books", "balance the {D} transfers with accounting records"]),
    ("pay", ["integrate webhooks from {D}", "subscribe to event notifications coming from {D}", "handle callbacks sent by {D}", "receive and verify push events from {D}"]),
    ("ci", ["speed up slow pipelines on {D}", "make {D} builds finish faster", "shorten our {D} run times with caching", "cut the duration of jobs running in {D}"]),
    ("ci", ["store and rotate secrets in {D}", "manage secret variables on {D}", "keep tokens safe in {D} jobs", "handle confidential values for {D} workflows"]),
    ("msg", ["handle poison messages in {D}", "deal with messages that keep failing in {D}", "set up a dead-letter path on {D}", "quarantine unprocessable events on {D}"]),
    ("msg", ["monitor consumer lag on {D}", "watch for backlog growth in {D}", "alert when readers fall behind on {D}", "track how far consumers trail producers on {D}"]),
    ("plat", ["submit a new release to the {D} store", "publish an update for the {D} app", "push a build through {D} review", "get our latest {D} version approved and listed"]),
    ("plat", ["debug crashes reported on {D}", "investigate app crash reports from {D}", "fix crashing behaviour on {D}", "trace the cause of failures users see on {D}"]),
    ("fmt", ["import customer data from {D} files", "load records out of {D} exports", "ingest {D} uploads", "read bulk customer records supplied as {D}"]),
    ("fmt", ["validate the schema of incoming {D} data", "check the structure of {D} inputs", "reject malformed {D} documents", "verify that {D} files match the expected layout"]),
    ("reg", ["migrate user data into the {D} region for compliance", "move customer records to {D} data residency", "relocate personal data into {D}", "keep personal information inside the {D} region"]),
    ("reg", ["pick a CDN and latency budget for users in {D}", "tune page load speed for visitors from {D}", "reduce latency for customers in {D}", "set response-time targets for people in {D}"]),
]

FILLER = [
    "Start by confirming you have the necessary access and a recent snapshot.",
    "Notify the on-call engineer before you begin.",
    "Record each action in the change log for auditing.",
    "Verify the result with a smoke test once finished.",
    "If something looks wrong, roll back and escalate to the owning team.",
    "Allow about an hour and avoid peak traffic windows.",
    "Check the dashboard for alerts before and after the change.",
    "Ask a colleague to review the plan if you have not done this before.",
    "Keep a copy of the previous configuration in case you need it.",
    "Document anything unexpected so the guide can be improved.",
    "Make sure the maintenance window has been announced in the team channel.",
    "Do not skip the validation step even when the run looks clean.",
    "Follow the least-privilege principle and use your personal account.",
    "Save the output of every command for the ticket.",
    "Contact support only after the basic checks have been completed.",
    "The estimated effort is small, but the consequences of mistakes are large.",
]
DETAIL_SENT = [
    "The steps below assume you are working with {D}.",
    "If you are not using {D}, this procedure does not apply.",
    "Settings that are specific to {D} are listed at the end of the page.",
    "Pay special attention to how {D} differs from the other options.",
    "Everything here is written for {D} and nothing else.",
    "Double-check that the target really is {D} before you run anything.",
]
QUERY_PAT = ["How do I {a}?", "What is the procedure to {a}?", "Need help: I have to {a}.", "Can someone tell me how to {a}?",
             "Looking for instructions on how to {a}", "Where are the steps to {a}?", "I want to {a}, how should I go about it?"]
PREAMBLE = ["Hey team, quick question after the meeting yesterday and sorry if this was asked before,",
            "Hi all, I know it is Friday afternoon and everybody is busy, but",
            "So my manager keeps pinging me about this and I am a bit lost, could you tell me",
            "Not urgent at all, just curious while I wait for my coffee to finish:"]
POSTAMBLE = ["Thanks a lot in advance!", "Any pointer helps, cheers.", "Happy to jump on a call if easier.", "Appreciate it."]
STOP = set("a an the to of in on for and or is are be do does how what where i we you it my our your can could should would that this with from at as by about any me us".split())

OFF_SUBJ = {
    "cooking": ["sourdough bread", "tomato soup", "vegetable lasagne", "chocolate brownies", "chicken curry", "pancakes", "risotto", "apple pie", "guacamole", "ramen broth"],
    "travel": ["a weekend in Lisbon", "a hiking trip in Norway", "a road trip through Utah", "a train tour of Japan", "a cruise around Greece", "a week in Marrakech", "a camping holiday in Scotland", "a city break in Prague"],
    "garden": ["tomato plants", "a herb garden", "rose bushes", "a lawn after winter", "compost heaps", "orchids", "raised vegetable beds", "a bonsai tree"],
    "fitness": ["a half marathon", "a pull-up routine", "stretching after cycling", "a strength plan for beginners", "swimming technique", "a yoga morning routine", "a 5k run"],
    "movies": ["a classic film noir", "an animated family film", "a sci-fi trilogy", "a documentary about oceans", "a silent comedy", "a heist thriller"],
}
OFF_DOC = ["Here is a simple way to prepare {s}.", "Many people ask how to get better results with {s}.", "A short guide about {s} for first-timers.",
           "Common mistakes with {s} and how to avoid them.", "Our community shares favourite tips on {s} every week.",
           "Whether you are experienced or new, {s} rewards patience.", "Budget a little extra time for {s} the first time.",
           "Friends and family usually love the outcome of {s}.", "Read the reviews before you commit to {s}.", "Take notes so that next time {s} goes even smoother."]
OFF_Q = ["What is the best way to get started with {s}?", "Any tips for {s}?", "How do beginners usually handle {s}?", "I would love advice about {s}.", "Where can I read more on {s}?"]


def typo_word(w, r):
    if len(w) < 5 or not w.isalpha():
        return w
    i = r.randrange(1, len(w) - 1)
    op = r.randrange(4)
    if op == 0:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == 1:
        return w[:i] + w[i + 1:]
    if op == 2:
        return w[:i] + w[i] + w[i:]
    kb = "qwertyuiopasdfghjklzxcvbnm"
    return w[:i] + kb[(kb.index(w[i].lower()) + r.choice([-1, 1])) % 26 if w[i].lower() in kb else 0] + w[i + 1:]


def add_typos(text, r, n):
    toks = re.findall(r"\w+|\W+", text)
    idx = [i for i, t in enumerate(toks) if t.isalpha() and len(t) >= 5]
    r.shuffle(idx)
    for i in idx[:n]:
        toks[i] = typo_word(toks[i], r)
    return "".join(toks)


def words(s):
    return re.findall(r"[a-z0-9\+#\.\-/']+", s.lower())


def edits1(w):
    letters = "abcdefghijklmnopqrstuvwxyz"
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = {a + b[1:] for a, b in sp if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1}
    out |= {a + c + b[1:] for a, b in sp if b for c in letters}
    out |= {a + c + b for a, b in sp for c in letters}
    return out


def build(seed):
    r = random.Random(seed * 1000003 + 17)
    concepts, docs, queries = [], [], []
    for ti, (pool, acts) in enumerate(TOPICS):
        for di, forms in enumerate(POOLS[pool]):
            concepts.append(dict(id=len(concepts), topic=ti, detail=di, pool=pool, forms=list(forms), acts=acts))
    nconc = len(concepts)
    for c in concepts:
        forms = c["forms"][:]
        if seed != 1:
            r.shuffle(forms)  # which surface form is f0/f1/f2 differs between corpora
        acts = c["acts"][:]
        r.shuffle(acts)
        fill = r.sample(FILLER, 4)
        ds = r.sample(DETAIL_SENT, 2)
        a_pair, b_pair = (acts[0], acts[1]), (acts[2], acts[3])
        specs = [(a_pair, (forms[0], forms[1], forms[0]), fill[:2]), (b_pair, (forms[1], forms[2], forms[2]), fill[2:])]
        for k, (ap, fm, fl) in enumerate(specs):
            sents = ["How to %s." % ap[0].format(D=fm[0]), "This runbook explains how to %s, step by step." % ap[1].format(D=fm[1]),
                     fl[0], ds[k].format(D=fm[2]), fl[1]]
            if r.random() < 0.5:
                sents[2], sents[3] = sents[3], sents[2]
            docs.append(dict(id=len(docs), concept=c["id"], topic=c["topic"], kind="doc", sents=sents, text=" ".join(sents)))
        for qi in range(8):
            act = r.choice(c["acts"]).format(D=r.choices(c["forms"], weights=[0.35, 0.35, 0.30])[0])
            if qi < 3:
                cls, t = "clean", r.choice(QUERY_PAT).format(a=act)
            elif qi < 5:
                cls, t = "typo", add_typos(r.choice(QUERY_PAT).format(a=act), r, r.choice([1, 2, 3]))
            elif qi < 7:
                cls = "terse"
                form = r.choices(c["forms"], weights=[0.35, 0.35, 0.30])[0]
                cw = [w for w in words(r.choice(c["acts"]).replace("{D}", " ")) if w not in STOP]
                kws = r.sample(cw, min(3, len(cw))) + [form]
                r.shuffle(kws)
                t = " ".join(kws)
            else:
                cls = "chatty"
                t = "%s %s %s" % (r.choice(PREAMBLE), "how do I %s?" % act, r.choice(POSTAMBLE))
            queries.append(dict(id=len(queries), concept=c["id"], topic=c["topic"], cls=cls, text=t))
    # off-topic docs + OOD queries
    subs = [(k, s) for k, v in OFF_SUBJ.items() for s in v]
    for i in range(300):
        k, s = subs[i % len(subs)]
        sents = [x.format(s=s) for x in r.sample(OFF_DOC, 4)]
        docs.append(dict(id=len(docs), concept=-1, topic=-1, kind="off", sents=sents, text=" ".join(sents)))
    for i in range(200):
        k, s = subs[r.randrange(len(subs))]
        t = r.choice(OFF_Q).format(s=s)
        if r.random() < 0.3:
            t = add_typos(t, r, 1)
        queries.append(dict(id=len(queries), concept=-1, topic=-1, cls="ood", text=t))
    # rewrites from DOC vocabulary only
    freq = collections.Counter(w for d in docs for w in words(d["text"]))
    qcnt = collections.Counter(w for q in queries for w in set(words(q["text"])))
    known = set(freq) | {w for w, c in qcnt.items() if c >= 3}  # doc vocab + words recurring in the query log (typos do not recur)
    df = collections.Counter(w for d in docs for w in set(words(d["text"])))
    N = len(docs)

    def correct(w):
        if w in known or not w.isalpha() or len(w) < 4:
            return w
        c1 = [x for x in edits1(w) if x in freq]
        if c1:
            return max(c1, key=lambda x: freq[x])
        return w

    for q in queries:
        toks = re.findall(r"\w+|\W+", q["text"])
        sp = "".join(correct(t.lower()) if t.isalpha() else t for t in toks)
        q["spell"] = sp
        kw = [w for w in words(sp) if w in df and df[w] / N <= 0.25]
        q["kw"] = " ".join(kw) if kw else sp
    return dict(seed=seed, n_concepts=nconc, n_topics=len(TOPICS), concepts=[{k: v for k, v in c.items() if k != "acts"} for c in concepts],
                docs=docs, queries=queries)


if __name__ == "__main__":
    seed, out = int(sys.argv[1]), sys.argv[2]
    d = build(seed)
    json.dump(d, open(out, "w", encoding="utf8"), indent=0)
    nq = collections.Counter(q["cls"] for q in d["queries"])
    print("seed", seed, "concepts", d["n_concepts"], "docs", len(d["docs"]), "queries", dict(nq))
    for q in d["queries"][:8]:
        print(" ", q["cls"], "|", q["text"], "|", q["spell"], "|", q["kw"])
    print("  doc0:", d["docs"][0]["text"])
    print("  doc1:", d["docs"][1]["text"])
