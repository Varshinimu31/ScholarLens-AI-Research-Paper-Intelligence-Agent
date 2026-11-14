import os, re, math, time, uuid, requests, xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Research Paper Intelligence Agent", version="1.0.0", description="Academic search, literature review and grounded research Q&A")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
HEADERS={"User-Agent":"ResearchPaperIntelligenceAgent/1.0 (academic portfolio project)"}
PAPERS={}
LAST_SEARCH=[]
STOP=set('the a an and or of for in on with to from by is are as at that this these those we our their using use study paper research results method methods model models based'.split())

def plain(s): return re.sub(r'\s+',' ',s or '').strip()
def paper_id(source, raw): return source+':'+str(raw)
def clean_paper(p):
 return {"id":p['id'],"title":p['title'],"authors":p.get('authors',[]),"year":p.get('year'),"abstract":p.get('abstract',''),"url":p.get('url',''),"source":p.get('source',''),"topics":p.get('topics',[]),"published":p.get('published','')}
def fetch_arxiv(query, limit):
 url='https://export.arxiv.org/api/query'
 response=requests.get(url,params={'search_query':'all:'+query,'start':0,'max_results':limit,'sortBy':'relevance','sortOrder':'descending'},headers=HEADERS,timeout=22)
 response.raise_for_status()
 ns={'a':'http://www.w3.org/2005/Atom'}
 root=ET.fromstring(response.content)
 out=[]
 for e in root.findall('a:entry',ns):
  link=plain(e.findtext('a:id','',ns)); title=plain(e.findtext('a:title','',ns)); summary=plain(e.findtext('a:summary','',ns)); published=plain(e.findtext('a:published','',ns))
  authors=[plain(x.findtext('a:name','',ns)) for x in e.findall('a:author',ns)]
  categories=[x.attrib.get('term','') for x in e.findall('{http://arxiv.org/schemas/atom}primary_category')]
  out.append(clean_paper({'id':paper_id('arxiv',link.split('/')[-1]),'title':title,'authors':authors,'year':int(published[:4]) if published[:4].isdigit() else None,'abstract':summary,'url':link,'source':'arXiv','topics':categories,'published':published[:10]}))
 return out

def reconstruct_abstract(index):
 if not index: return ''
 words=sorted(((pos,word) for word,positions in index.items() for pos in positions))
 return plain(' '.join(w for _,w in words))

def fetch_openalex(query, limit):
 params={'search':query,'per-page':limit,'select':'id,title,authorships,publication_year,publication_date,abstract_inverted_index,primary_location,primary_topic,doi'}
 email=os.getenv('OPENALEX_EMAIL','').strip()
 if email: params['mailto']=email
 response=requests.get('https://api.openalex.org/works',params=params,headers=HEADERS,timeout=22)
 response.raise_for_status()
 out=[]
 for x in response.json().get('results',[]):
  title=plain(x.get('title'))
  if not title:continue
  location=x.get('primary_location') or {}
  url=x.get('doi') or location.get('landing_page_url') or x.get('id','')
  topic=(x.get('primary_topic') or {}).get('display_name')
  out.append(clean_paper({'id':paper_id('openalex',x.get('id','').rsplit('/',1)[-1]),'title':title,'authors':[a.get('author',{}).get('display_name','') for a in x.get('authorships',[]) if a.get('author')],'year':x.get('publication_year'),'abstract':reconstruct_abstract(x.get('abstract_inverted_index')),'url':url,'source':'OpenAlex','topics':[topic] if topic else [],'published':x.get('publication_date') or ''}))
 return out

def tokenize(s): return [w for w in re.findall(r'[a-z0-9]{3,}',s.lower()) if w not in STOP]
def lexical_rank(question,papers):
 q=Counter(tokenize(question)); ranked=[]
 for p in papers:
  terms=Counter(tokenize(p['title']+' '+p.get('abstract','')))
  score=sum(q[w]*min(terms[w],4) for w in q)
  ranked.append((score,p))
 return sorted(ranked,key=lambda x:x[0],reverse=True)

def retrieve(question,papers,k):
 """Semantic retrieval when optional sentence-transformers installed; lexical fallback otherwise."""
 if os.getenv('USE_EMBEDDINGS','0')=='1':
  try:
   from sentence_transformers import SentenceTransformer
   import numpy as np
   global EMBEDDER
   if EMBEDDER is None: EMBEDDER=SentenceTransformer('all-MiniLM-L6-v2')
   texts=[p['title']+'\n'+p.get('abstract','') for p in papers]
   vectors=EMBEDDER.encode([question]+texts,normalize_embeddings=True)
   scores=np.dot(vectors[1:],vectors[0])
   return [(float(scores[i]),papers[i]) for i in np.argsort(-scores)[:k]],'semantic'
  except Exception as e:
   print('Embedding unavailable, lexical fallback:',e)
 return lexical_rank(question,papers)[:k],'lexical'
EMBEDDER=None

def llm(prompt):
 model=os.getenv('OLLAMA_MODEL','llama3.2:3b')
 url=os.getenv('OLLAMA_URL','http://127.0.0.1:11434').rstrip('/')+'/api/generate'
 try:
  response=requests.post(url,json={'model':model,'prompt':prompt,'stream':False,'options':{'temperature':0.2}},timeout=150)
  response.raise_for_status()
  return response.json().get('response','').strip()
 except requests.RequestException as e:
  raise HTTPException(503,detail='Ollama is unavailable. Start Ollama and pull '+model+'. Details: '+str(e))

def get_papers(ids):
 missing=[i for i in ids if i not in PAPERS]
 if missing: raise HTTPException(404,detail='Paper IDs not found in this server session: '+', '.join(missing[:3])+'. Search again.')
 return [PAPERS[i] for i in ids]

def evidence(papers):
 return '\n\n'.join('[%d] %s (%s). %s\nAuthors: %s\nAbstract: %s\nURL: %s' % (i,p['title'],p.get('year') or 'n.d.',p['source'],', '.join(p['authors'][:5]),p.get('abstract') or '[No abstract available]',p['url']) for i,p in enumerate(papers,1))

class PaperIDs(BaseModel): paper_ids:list[str]=Field(min_length=1,max_length=12)
class AskRequest(PaperIDs): question:str=Field(min_length=3,max_length=1000); top_k:int=Field(default=5,ge=1,le=10)
class SearchResponse(BaseModel): papers:list[dict]; total:int; errors:list[str]

@app.get('/')
def home():return {'name':'Research Paper Intelligence Agent','status':'running','docs':'/docs'}
@app.get('/health')
def health():return {'status':'ok','papers_cached':len(PAPERS),'ollama_model':os.getenv('OLLAMA_MODEL','llama3.2:3b')}
@app.get('/papers/search',response_model=SearchResponse)
def search(query:str=Query(min_length=2,max_length=200),source:str=Query('both',pattern='^(both|arxiv|openalex)$'),limit:int=Query(8,ge=1,le=20),year:Optional[int]=None,author:Optional[str]=None):
 global LAST_SEARCH
 found=[];errors=[]
 for name,fn in [('arxiv',fetch_arxiv),('openalex',fetch_openalex)]:
  if source not in ('both',name):continue
  try:found.extend(fn(query,limit))
  except Exception as e:errors.append(f'{name}: {str(e)}')
 if not found and errors:raise HTTPException(502,detail='Academic providers unavailable: '+'; '.join(errors))
 dedup={};
 for p in found:
  key=re.sub(r'[^a-z0-9]','',p['title'].lower())
  if key not in dedup or (not dedup[key]['abstract'] and p['abstract']):dedup[key]=p
 filtered=list(dedup.values())
 if year:filtered=[p for p in filtered if p['year']==year]
 if author:filtered=[p for p in filtered if author.lower() in ' '.join(p['authors']).lower()]
 for p in filtered:PAPERS[p['id']]=p
 LAST_SEARCH=[p['id'] for p in filtered]
 return {'papers':filtered,'total':len(filtered),'errors':errors}
@app.get('/papers/{paper_id:path}')
def details(paper_id:str):
 if paper_id not in PAPERS:raise HTTPException(404,detail='Paper not in session. Search first.')
 return PAPERS[paper_id]
@app.post('/papers/summarize')
def summarize(body:PaperIDs):
 papers=get_papers(body.paper_ids)
 prompt='Summarize each paper separately using ONLY its abstract. Include objective, approach (only if stated), findings (only if stated), limitations or missing details, and [n] citations. Never invent full-text details.\n\n'+evidence(papers)
 return {'analysis':llm(prompt),'citations':[{'number':i,'title':p['title'],'url':p['url'],'id':p['id']} for i,p in enumerate(papers,1)]}
@app.post('/papers/compare')
def compare(body:PaperIDs):
 papers=get_papers(body.paper_ids)
 if len(papers)<2:raise HTTPException(422,detail='Select at least two papers to compare.')
 prompt='Compare the research methods, results, similarities, differences and limitations based ONLY on the supplied abstracts. Use a concise Markdown comparison table and cite [n]. If methodology is not stated, explicitly say so.\n\n'+evidence(papers)
 return {'analysis':llm(prompt),'citations':[{'number':i,'title':p['title'],'url':p['url'],'id':p['id']} for i,p in enumerate(papers,1)]}
@app.post('/research/gaps')
def gaps(body:PaperIDs):
 papers=get_papers(body.paper_ids)
 prompt='Identify possible research gaps and promising future directions grounded ONLY in these abstracts. Distinguish explicit limitations from your hypotheses, avoid claiming a gap is novel, cite [n] for each observation.\n\n'+evidence(papers)
 return {'analysis':llm(prompt),'citations':[{'number':i,'title':p['title'],'url':p['url'],'id':p['id']} for i,p in enumerate(papers,1)]}
@app.post('/research/ask')
def ask(body:AskRequest):
 papers=get_papers(body.paper_ids)
 ranked,mode=retrieve(body.question,papers,min(body.top_k,len(papers)))
 chosen=[p for score,p in ranked]
 prompt='Answer the question ONLY using the paper abstracts below. Cite supporting claims with [n]. If evidence is missing, say so. Do not invent details.\nQuestion: '+body.question+'\n\n'+evidence(chosen)
 return {'answer':llm(prompt),'retrieval_mode':mode,'citations':[{'number':i,'title':p['title'],'url':p['url'],'id':p['id'],'score':round(float(ranked[i-1][0]),3)} for i,p in enumerate(chosen,1)]}
