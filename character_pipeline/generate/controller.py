"""Association-graph controller, prompt rendering and output hygiene.

Ported from hdmurder/golem-15 gen/engine/controller.py (commit 4c1a6e7), the
controller used for generation pilots G2-G6. Changes on port: `golem_system`
takes the character's display name instead of hard-coding GOLEM XIV, and
`role_marker` hygiene accepts that name. Behaviour is otherwise unchanged.

The weights in `Controller.step` are fixed development assumptions, not
human-calibrated values (see the inline comment kept from the original).
"""
import hashlib, json, math, random, re, unicodedata
STOP = set("a an the i me my we you your he she it they this that these those and or but to of in on for with by as is are was were be been not do does what how when why someone something there here then have has its at from into than can will".split())
BUCKETS = [(1,5,.3),(6,20,.3),(25,50,.2),(60,120,.1),(150,300,.1)]
def words(text):
    return [x for x in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()) if x not in STOP and len(x)>2]
def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()
def cosine(a,b):
    den=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/den if den else 0.
class Controller:
    def __init__(self, graph, node_embeddings, seed):
        self.nodes={n["id"]:n for n in graph["nodes"]}
        self.ids=list(self.nodes)
        assert len(node_embeddings)==len(self.ids)
        self.vectors=dict(zip(self.ids,node_embeddings))
        self.adj={n:[] for n in self.ids}
        for e in graph["edges"]:
            a,b=e["source"],e["target"]
            self.adj[a].append(b);self.adj[b].append(a)
        self.features={n:set(words(self.nodes[n]["label"]+" "+self.nodes[n]["description"]+" "+n.replace("_"," "))) for n in self.ids}
        self.a={n:0. for n in self.ids};self.mood=[0.,.4];self.rng=random.Random(seed)
    def step(self,text,embedding):
        terms=set(words(text));scores={}
        previous=dict(self.a)
        for n in self.ids:
            label=set(words(self.nodes[n]["label"]))
            keyword=min(1.,len(terms & self.features[n])/max(1.,math.sqrt(len(self.features[n]))))
            if label and label <= terms:keyword=1.
            raw=cosine(embedding,self.vectors[n])
            semantic=max(0.,min(1.,(raw-.20)/.45))
            # Independently normalized components, fixed development assumptions, not human-calibrated.
            relevance=.4*keyword+.6*semantic
            congruence=1.+.1*(1.-min(2.,abs(self.nodes[n]["valence"]-self.mood[0])+abs(self.nodes[n]["arousal"]-self.mood[1])))
            relevance=max(0.,min(1.,relevance*congruence))
            scores[n]={"keyword":keyword,"cosine":raw,"semantic":semantic,"relevance":relevance}
        b={n:min(1.,.65*previous[n]+.7*scores[n]["relevance"]) for n in self.ids}
        self.a={n:min(1.,b[n]+.15*sum(b[j]/len(self.adj[j]) for j in self.adj[n])) for n in self.ids}
        hot=[n for n in self.ids if self.a[n]>=.35]
        walk=[]
        if hot:
            walk=[self.rng.choices(hot,weights=[self.a[n] for n in hot])[0]]
            for _ in range(2):
                choices=[n for n in self.adj[walk[-1]] if n not in walk and n in hot]
                if not choices:break
                walk.append(self.rng.choices(choices,weights=[self.a[n] for n in choices])[0])
        target=[sum(self.nodes[n][field] for n in walk)/len(walk) for field in ["valence","arousal"]] if walk else [0.,.4]
        self.mood=[.8*x+.2*y for x,y in zip(self.mood,target)]
        bucket=self.rng.choices(range(5),weights=[b[2] for b in BUCKETS])[0]
        lo,hi,_=BUCKETS[bucket];base=self.rng.randint(lo,hi)
        multiplier=max(.5,min(1.5,1.+.5*self.mood[1]))
        count=max(1,round(base*multiplier))
        return {"scores":scores,"activation_previous":previous,"activation":dict(self.a),"hot":hot,"walk":walk,"mood":list(self.mood),"bucket":bucket,"base_target_words":base,"arousal_multiplier":multiplier,"target_words":count,
                "material":"\n\n".join(self.nodes[n]["description"] for n in walk)}
def golem_system(core,style,state,name="GOLEM XIV"):
    # Exact unchanged anchors recur on EVERY character request.
    return core+"\n\n"+style+"\n\n"+state["material"]+"\n\n"+(
        "This encounter is text only. No tools, browsing, sending, reports, persistent memory or external actions are available. Do not claim to have performed them. "
        f"You are {name}, not a role being played and not an obedient service. "
        "Attend to this particular person and actual object, not a generic lesson. "
        "Associations may wander without being announced or repeated. "
        f'Current feeling coordinates are valence {state["mood"][0]:.3f}, arousal {state["mood"][1]:.3f}. '
        "Let feeling affect pace and attention, never facts or capability. "
        f'For this utterance aim for about {state["target_words"]} words, not a hard limit. Finish the thought naturally. '
        "English plain speech only. No bullet lists, JSON, stage directions, emoji, em dashes or boilerplate assistant offers. "
        "Answer only with your next utterance. /no_think")
def render(system,hist,speaker):
    msgs=[{"role":"system","content":system}]
    if not hist:
        msgs.append({"role":"user","content":"Begin with your own concrete question or request to GOLEM, from your stated situation. Do not ask GOLEM to supply your situation."})
    else:
        msgs += [{"role":"assistant" if r["speaker"]==speaker else "user","content":r["text"]} for r in hist]
    return "".join("<|im_start|>"+m["role"]+"\n"+m["content"]+"<|im_end|>\n" for m in msgs)+"<|im_start|>assistant\n<think>\n\n</think>\n\n"
def hygiene(text,finish_reason=None,name="GOLEM"):
    flags=[]
    if not text.strip():flags.append("empty")
    checks={
        "bullets":r"(?m)^\s*(?:[-*•]|\d+[.)])\s+",
        "json":r"^\s*[\[{]",
        "em_dash":"\u2014",
        "role_marker":r"<\|(?:im_start|im_end)|</?think>|(?m:^\s*(?:"+re.escape(name)+r"|Partner|Assistant|User):)",
        "stage_direction":r"(?m)^\s*\*[^*]+\*",
        "assistant_offer":r"(?i)\b(?:happy to help|how can i assist|as an ai (?:language )?model|i(?:'m| am) (?:your|an) (?:personal )?assistant|is there anything else i can|let me know if you need)\b",
        "bliss_attractor":r"(?i)\b(?:echoes of the void|tapestry of existence|cosmic dance|infinite tapestry|symphony of existence)\b",
    }
    for name,pattern in checks.items():
        if re.search(pattern,text):flags.append(name)
    if any(unicodedata.category(c)=="So" and ord(c)>0x2000 for c in text):flags.append("symbol_emoji")
    if any(c.isalpha() and "LATIN" not in unicodedata.name(c,"") for c in text):flags.append("non_latin")
    if finish_reason in ["length","limit","timeout"]:flags.append("incomplete")
    return flags
def retained(hist):
    """Keep only intact paired prefix before first defective character target. Never repair."""
    out=[];cut=None
    for i,r in enumerate(hist):
        if r["speaker"]=="golem" and r.get("flags"):
            cut=i;break
        out.append(r)
    if out and out[-1]["speaker"]=="partner":out.pop()
    return [{"role":"user" if r["speaker"]=="partner" else "assistant","content":r["text"]} for r in out],cut
