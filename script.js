// ════════════════════════════════════════════════════════════
//  CONSTANTS
// ════════════════════════════════════════════════════════════
const API  = 'http://localhost:8080';
const DIMS = 16;
const COL  = { cs:'#00d9ff', math:'#b388ff', food:'#ffb74d', sports:'#69f0ae', doc:'#a6e3a1', default:'#90a4ae' };
const DIM_COL = ['#00d9ff','#00d9ff','#00d9ff','#00d9ff','#b388ff','#b388ff','#b388ff','#b388ff',
                 '#ffb74d','#ffb74d','#ffb74d','#ffb74d','#69f0ae','#69f0ae','#69f0ae','#69f0ae'];

// ════════════════════════════════════════════════════════════
//  STATE
// ════════════════════════════════════════════════════════════
let allItems = [], pcaPoints = [], hitIds = new Set(), queryPt = null;
let hoverItem = null, pulse = 0, selAlgo = 'hnsw', searchResults = [];

// ════════════════════════════════════════════════════════════
//  TABS
// ════════════════════════════════════════════════════════════
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const names = ['search','docs','rag'];
    t.classList.toggle('on', names[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('on'));
  document.getElementById('tab-' + name).classList.add('on');
  if (name === 'docs') loadDocList();
}

// ════════════════════════════════════════════════════════════
//  TEXT → 16-D EMBEDDING (for demo vectors)
// ════════════════════════════════════════════════════════════
const KW = {
  cs:     ['algorithm','data','tree','graph','array','linked','hash','stack','queue','sort','binary','dynamic','programming','recursion','complexity','pointer','node','search','insert','bfs','dfs','heap','trie'],
  math:   ['calculus','matrix','probability','theorem','integral','derivative','linear','algebra','equation','function','prime','modular','combinatorics','permutation','eigenvalue','statistics','proof'],
  food:   ['food','pizza','sushi','ramen','pasta','recipe','cook','eat','restaurant','dish','ingredient','flavor','spice','noodle','bread','croissant','taco','fish','rice','soup'],
  sports: ['sport','basketball','football','tennis','chess','swim','game','play','score','team','athlete','competition','match','tournament','olympic','dribble','tackle','serve']
};

function textToEmbedding(text) {
  const t = text.toLowerCase(), ws = t.split(/\s+/);
  const s = {cs:0,math:0,food:0,sports:0};
  for (const w of ws)
    for (const [cat, kws] of Object.entries(KW))
      for (const kw of kws) if (w.includes(kw)||kw.startsWith(w)) { s[cat]+=0.35; break; }
  const mx = Math.max(...Object.values(s), 0.01);
  const n = v => Math.min(v/mx*0.88, 0.94);
  const jitter = () => (Math.random()-.5)*.04;
  const emb = new Array(16).fill(0.08);
  const fill = (i,score) => {
    if (score<.01) return;
    const b = n(score);
    emb[i]=Math.max(.05,b+jitter()); emb[i+1]=Math.max(.05,b+jitter());
    emb[i+2]=Math.max(.05,b*.92+jitter()); emb[i+3]=Math.max(.05,b*.87+jitter());
  };
  fill(0,s.cs); fill(4,s.math); fill(8,s.food); fill(12,s.sports);
  return emb;
}

// ════════════════════════════════════════════════════════════
//  PCA
// ════════════════════════════════════════════════════════════
function pca2D(embs) {
  const n = embs.length, d = embs[0].length;
  if (n < 2) return embs.map(() => [0,0]);
  const mean = new Array(d).fill(0);
  for (const e of embs) for (let i=0;i<d;i++) mean[i]+=e[i]/n;
  const X = embs.map(e => e.map((v,i)=>v-mean[i]));
  function powerIter(X,excl) {
    let v = new Array(d).fill(0).map(()=>Math.random()-.5);
    if (excl) { let dot=v.reduce((s,vi,i)=>s+vi*excl[i],0); v=v.map((vi,i)=>vi-dot*excl[i]); }
    let nrm = Math.sqrt(v.reduce((s,vi)=>s+vi*vi,0));
    v = v.map(vi=>vi/nrm);
    for (let it=0;it<200;it++) {
      const Xv=X.map(xi=>xi.reduce((s,xij,j)=>s+xij*v[j],0));
      const nv=new Array(d).fill(0);
      for (let k=0;k<n;k++) for (let j=0;j<d;j++) nv[j]+=X[k][j]*Xv[k];
      if (excl) { let dot=nv.reduce((s,vi,i)=>s+vi*excl[i],0); for (let i=0;i<d;i++) nv[i]-=dot*excl[i]; }
      nrm=Math.sqrt(nv.reduce((s,vi)=>s+vi*vi,0));
      if (nrm<1e-10) break;
      const prev=v.slice(); v=nv.map(vi=>vi/nrm);
      if (v.reduce((s,vi,i)=>s+(vi-prev[i])**2,0)<1e-12) break;
    }
    return v;
  }
  const pc1=powerIter(X,null), pc2=powerIter(X,pc1);
  return X.map(x=>[x.reduce((s,v,i)=>s+v*pc1[i],0),x.reduce((s,v,i)=>s+v*pc2[i],0)]);
}

// ════════════════════════════════════════════════════════════
//  SCATTER PLOT
// ════════════════════════════════════════════════════════════
const sc=document.getElementById('scatter'), ctx=sc.getContext('2d');
let bounds={minX:-1,maxX:1,minY:-1,maxY:1};

function resize() { const r=sc.parentElement.getBoundingClientRect(); sc.width=r.width; sc.height=r.height; }
window.addEventListener('resize', resize);

function w2c(wx,wy) {
  const P=70,W=sc.width,H=sc.height,rx=bounds.maxX-bounds.minX||1,ry=bounds.maxY-bounds.minY||1;
  return [P+((wx-bounds.minX)/rx)*(W-2*P), H-P-((wy-bounds.minY)/ry)*(H-2*P)];
}

function drawFrame() {
  ctx.clearRect(0,0,sc.width,sc.height);
  ctx.fillStyle='#07070f'; ctx.fillRect(0,0,sc.width,sc.height);
  ctx.strokeStyle='#0e0e1e'; ctx.lineWidth=1;
  for (let i=0;i<=8;i++) {
    const tx=70+(i/8)*(sc.width-140),ty=70+(i/8)*(sc.height-140);
    ctx.beginPath();ctx.moveTo(tx,70);ctx.lineTo(tx,sc.height-70);ctx.stroke();
    ctx.beginPath();ctx.moveTo(70,ty);ctx.lineTo(sc.width-70,ty);ctx.stroke();
  }
  ctx.fillStyle='#1a1a38'; ctx.font='11px Fira Code,monospace';
  ctx.fillText('PC₁ →',sc.width/2-40,sc.height-18);
  ctx.save();ctx.translate(18,sc.height/2+50);ctx.rotate(-Math.PI/2);ctx.fillText('PC₂ →',0,0);ctx.restore();
  ctx.fillStyle='#151530'; ctx.font='12px Fira Code,monospace';
  ctx.fillText('2D PCA Projection  ·  Semantic Space',80,28);

  if (queryPt && hitIds.size>0) {
    const [qx,qy]=w2c(queryPt.x,queryPt.y);
    for (const pt of pcaPoints) {
      if (!hitIds.has(pt.item.id)) continue;
      const [px,py]=w2c(pt.x,pt.y);
      ctx.strokeStyle='rgba(108,99,255,0.18)'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
      ctx.beginPath();ctx.moveTo(qx,qy);ctx.lineTo(px,py);ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  for (const pt of pcaPoints) {
    const [cx,cy]=w2c(pt.x,pt.y);
    const col=COL[pt.item.category]||COL.default;
    const isHit=hitIds.has(pt.item.id), r=isHit?10:7;
    if (isHit) {
      const pr=r+7+Math.sin(pulse)*3.5;
      ctx.beginPath();ctx.arc(cx,cy,pr,0,2*Math.PI);
      ctx.strokeStyle=col+'55';ctx.lineWidth=1.5;ctx.stroke();
    }
    const grd=ctx.createRadialGradient(cx,cy,0,cx,cy,r*3);
    grd.addColorStop(0,col+(isHit?'bb':'88'));grd.addColorStop(1,'transparent');
    ctx.beginPath();ctx.arc(cx,cy,r*3,0,2*Math.PI);ctx.fillStyle=grd;ctx.fill();
    ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);ctx.fillStyle=col;ctx.fill();
    if (hoverItem&&hoverItem.id===pt.item.id) {
      ctx.beginPath();ctx.arc(cx,cy,r+5,0,2*Math.PI);ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.stroke();
    }
  }
  if (queryPt) {
    const [qx,qy]=w2c(queryPt.x,queryPt.y);
    ctx.save();ctx.translate(qx,qy);
    ctx.shadowColor='#fff';ctx.shadowBlur=18;
    ctx.beginPath();
    for (let i=0;i<10;i++){const a=(i*Math.PI/5)-Math.PI/2,rr=i%2===0?13:5;if(i===0)ctx.moveTo(Math.cos(a)*rr,Math.sin(a)*rr);else ctx.lineTo(Math.cos(a)*rr,Math.sin(a)*rr);}
    ctx.closePath();ctx.fillStyle='#fff';ctx.fill();
    ctx.shadowBlur=0;ctx.restore();
    ctx.fillStyle='#aaaacc';ctx.font='10px Fira Code,monospace';ctx.fillText('query',qx+16,qy+4);
  }
  if (!pcaPoints.length) {
    ctx.fillStyle='#1a1a38';ctx.font='13px Fira Code,monospace';ctx.textAlign='center';
    ctx.fillText('Connecting to VectorDB…',sc.width/2,sc.height/2);ctx.textAlign='left';
  }
  pulse+=0.05;
  requestAnimationFrame(drawFrame);
}

sc.addEventListener('mousemove', e => {
  const rect=sc.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
  hoverItem=null; let best=18;
  for (const pt of pcaPoints) {
    const [cx,cy]=w2c(pt.x,pt.y),d=Math.hypot(mx-cx,my-cy);
    if (d<best){best=d;hoverItem=pt.item;}
  }
  const tip=document.getElementById('tip');
  if (hoverItem) {
    const col=COL[hoverItem.category]||COL.default;
    tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-8)+'px';
    tip.replaceChildren();
    const span = document.createElement('span');
    span.style.color = col;
    span.textContent = `[${hoverItem.category}]`;
    tip.appendChild(span);
    tip.appendChild(document.createElement('br'));
    tip.appendChild(document.createTextNode(hoverItem.metadata));
  } else tip.style.display='none';
});
sc.addEventListener('mouseleave',()=>{hoverItem=null;document.getElementById('tip').style.display='none';});

// ════════════════════════════════════════════════════════════
//  LOAD DEMO ITEMS
// ════════════════════════════════════════════════════════════
async function loadItems() {
  try {
    const r = await fetch(API+'/items');
    allItems = await r.json();
    if (allItems.length >= 2) {
      const coords = pca2D(allItems.map(v=>v.embedding));
      pcaPoints = allItems.map((item,i)=>({x:coords[i][0],y:coords[i][1],item}));
      let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
      for (const p of pcaPoints){x0=Math.min(x0,p.x);x1=Math.max(x1,p.x);y0=Math.min(y0,p.y);y1=Math.max(y1,p.y);}
      const px=(x1-x0)*.18||.1,py=(y1-y0)*.18||.1;
      bounds={minX:x0-px,maxX:x1+px,minY:y0-py,maxY:y1+py};
    }
    document.getElementById('statsLabel').textContent=allItems.length+' vectors · '+DIMS+' dims';
  } catch(_) {}
}

// ════════════════════════════════════════════════════════════
//  DEMO SEARCH
// ════════════════════════════════════════════════════════════
function setAlgo(el) {
  document.querySelectorAll('.algo-btn').forEach(b=>b.classList.remove('on'));
  el.classList.add('on'); selAlgo=el.dataset.algo;
}

async function runSearch() {
  const text=document.getElementById('qInput').value.trim(); if(!text)return;
  const emb=textToEmbedding(text),k=parseInt(document.getElementById('kSlider').value);
  const metric=document.getElementById('metric').value;
  const url=`${API}/search?v=${emb.join(',')}&k=${k}&metric=${metric}&algo=${selAlgo}`;
  try {
    const r=await fetch(url), data=await r.json();
    searchResults=data.results||[]; hitIds=new Set(searchResults.map(r=>r.id));
    const us=data.latencyUs||0;
    document.getElementById('latBig').textContent=us<1000?us+' μs':(us/1000).toFixed(2)+' ms';
    document.getElementById('latSub').textContent=selAlgo.toUpperCase()+'  ·  '+metric+'  ·  k='+k;
    if (searchResults.length>0){
      let sx=0,sy=0,sw=0;
      for (let i=0;i<Math.min(3,searchResults.length);i++){
        const pt=pcaPoints.find(p=>p.item.id===searchResults[i].id);
        if(pt){const w=1/(i+1);sx+=pt.x*w;sy+=pt.y*w;sw+=w;}
      }
      if(sw>0)queryPt={x:sx/sw+(Math.random()-.5)*.015,y:sy/sw+(Math.random()-.5)*.015};
    }
    renderResults(searchResults); drawVecChart(emb);
  } catch(_){alert('Cannot reach server — is it running on :8080?');}
}

document.getElementById('qInput').addEventListener('keydown',e=>{if(e.key==='Enter')runSearch();});

function renderResults(results) {
  const container = document.getElementById('results');
  container.replaceChildren();
  if (!results||!results.length){
    const div = document.createElement('div');
    div.style.color = 'var(--muted)'; div.style.fontSize = '11px'; div.textContent = 'No results';
    container.appendChild(div);
    return;
  }
  results.forEach((r,i)=>{
    const col=COL[r.category]||COL.default;
    const card = document.createElement('div'); card.className = 'rcard';
    card.onmouseenter = () => { hoverItem = {id: r.id}; };
    card.onmouseleave = () => { hoverItem = null; };
    const rrank = document.createElement('div'); rrank.className = 'rrank'; rrank.textContent = `#${i+1} NEAREST`;
    const rmeta = document.createElement('div'); rmeta.className = 'rmeta'; rmeta.textContent = r.metadata;
    const rfoot = document.createElement('div'); rfoot.className = 'rfoot';
    const rcat = document.createElement('span'); rcat.className = 'rcat';
    rcat.style.background = col+'18'; rcat.style.color = col; rcat.style.border = `1px solid ${col}44`;
    rcat.textContent = r.category.toUpperCase();
    const rdist = document.createElement('span'); rdist.className = 'rdist'; rdist.textContent = `dist: ${r.distance.toFixed(5)}`;
    const delBtn = document.createElement('button'); delBtn.className = 'del'; delBtn.textContent = '✕';
    delBtn.onclick = () => deleteItem(r.id);
    rfoot.append(rcat, rdist, delBtn);
    card.append(rrank, rmeta, rfoot);
    container.appendChild(card);
  });
}

function drawVecChart(emb) {
  const vc=document.getElementById('vecCvs'),W=vc.parentElement.clientWidth;
  vc.width=W;const vx=vc.getContext('2d');
  vx.clearRect(0,0,W,76);vx.fillStyle='#07070f';vx.fillRect(0,0,W,76);
  const bw=(W-4)/DIMS;
  for (let i=0;i<DIMS;i++){
    const h=emb[i]*58,x=2+i*bw,col=DIM_COL[i];
    vx.shadowColor=col;vx.shadowBlur=5;vx.fillStyle=col+'aa';vx.fillRect(x+1,63-h,bw-2,h);
  }
  vx.shadowBlur=0;vx.font='8px monospace';vx.textAlign='center';
  [['CS',0],['MATH',4],['FOOD',8],['SPORT',12]].forEach(([lbl,gi],i)=>{
    vx.fillStyle=Object.values(COL)[i]+'77';vx.fillText(lbl,2+(gi+1.5)*bw,74);
  });
  vx.textAlign='left';
}

async function runBenchmark() {
  const text=document.getElementById('qInput').value.trim()||'binary tree algorithm';
  const emb=textToEmbedding(text),metric=document.getElementById('metric').value;
  try {
    const r=await fetch(`${API}/benchmark?v=${emb.join(',')}&k=5&metric=${metric}`);
    const d=await r.json();
    document.getElementById('benchSec').style.display='block';
    const mx=Math.max(d.bruteforceUs,d.kdtreeUs,d.hnswUs,1);
    const container = document.getElementById('benchBars');
    container.replaceChildren();
    [
      {lbl:'Brute Force',us:d.bruteforceUs,col:'#f38ba8'},
      {lbl:'KD-Tree',    us:d.kdtreeUs,    col:'#89dceb'},
      {lbl:'HNSW',       us:d.hnswUs,      col:'#b388ff'},
    ].forEach(({lbl,us,col})=>{
      const pct=Math.max((us/mx)*100,2),disp=us<1000?us+' μs':(us/1000).toFixed(2)+' ms';
      const brow = document.createElement('div'); brow.className = 'brow';
      const blabel = document.createElement('div'); blabel.className = 'blabel';
      const span1 = document.createElement('span'); span1.style.color = col; span1.textContent = lbl;
      const span2 = document.createElement('span'); span2.style.color = 'var(--muted)'; span2.textContent = disp;
      blabel.append(span1, span2);
      const btrack = document.createElement('div'); btrack.className = 'btrack';
      const bfill = document.createElement('div'); bfill.className = 'bfill';
      bfill.style.width = pct+'%'; bfill.style.background = col;
      btrack.appendChild(bfill);
      brow.append(blabel, btrack);
      container.appendChild(brow);
    });
  } catch(_) {}
}

async function loadHNSW() {
  try {
    const r=await fetch(API+'/hnsw-info'), d=await r.json();
    const maxN=d.nodesPerLayer[0]||1;
    const container = document.getElementById('layers');
    container.replaceChildren();
    if (!d.nodesPerLayer || d.nodesPerLayer.length === 0) {
      const div = document.createElement('div'); div.style.color = 'var(--muted)'; div.style.fontSize = '11px'; div.textContent = 'Empty';
      container.appendChild(div);
      return;
    }
    d.nodesPerLayer.forEach((cnt,lyr)=>{
      const pct=Math.max((cnt/maxN)*100,2),edg=d.edgesPerLayer[lyr]||0;
      const lrow = document.createElement('div'); lrow.className = 'lrow';
      const lnum = document.createElement('div'); lnum.className = 'lnum'; lnum.textContent = `L${lyr}`;
      const ltrack = document.createElement('div'); ltrack.className = 'ltrack';
      const lfill = document.createElement('div'); lfill.className = 'lfill'; lfill.style.width = pct+'%';
      ltrack.appendChild(lfill);
      const lcount = document.createElement('div'); lcount.className = 'lcount'; lcount.textContent = `${cnt}n · ${edg}e`;
      lrow.append(lnum, ltrack, lcount);
      container.appendChild(lrow);
    });
  } catch(_) {}
}

async function addVector() {
  const meta=document.getElementById('addMeta').value.trim(),cat=document.getElementById('addCat').value;
  if(!meta)return;
  const emb=textToEmbedding(meta+' '+cat);
  try {
    await fetch(API+'/insert',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({metadata:meta,category:cat,embedding:emb})});
    document.getElementById('addMeta').value='';
    await loadItems();loadHNSW();
  } catch(_) {}
}

async function deleteItem(id) {
  try {
    await fetch(`${API}/delete/${id}`,{method:'DELETE'});
    searchResults=searchResults.filter(r=>r.id!==id);hitIds.delete(id);
    renderResults(searchResults);await loadItems();loadHNSW();
  } catch(_) {}
}

// ════════════════════════════════════════════════════════════
//  DOCUMENT MANAGEMENT
// ════════════════════════════════════════════════════════════
async function checkOllamaStatus() {
  try {
    const r=await fetch(API+'/status'), d=await r.json();
    const badge=document.getElementById('ollamaBadge');
    const box=document.getElementById('ollamaStatus');
    box.replaceChildren();
    if (d.ollamaAvailable) {
      badge.className='badge ok'; badge.textContent='OLLAMA ✓';
      box.className='ollama-status ok';
      const s1 = document.createElement('span'); s1.style.color='var(--green)'; s1.textContent='● Online';
      const br1 = document.createElement('br');
      const text1 = document.createTextNode('Embed: ');
      const s2 = document.createElement('span'); s2.style.color='var(--accent)'; s2.textContent=d.embedModel;
      const br2 = document.createElement('br');
      const text2 = document.createTextNode('Generate: ');
      const s3 = document.createElement('span'); s3.style.color='var(--accent)'; s3.textContent=d.genModel;
      const br3 = document.createElement('br');
      const text3 = document.createTextNode('Dims: ');
      const s4 = document.createElement('span'); s4.style.color='var(--muted)'; s4.textContent=d.docDims||'(first insert sets this)';
      const br4 = document.createElement('br');
      const text4 = document.createTextNode('Documents: ');
      const s5 = document.createElement('span'); s5.style.color='var(--text)'; s5.textContent=d.docCount;
      box.append(s1, br1, text1, s2, br2, text2, s3, br3, text3, s4, br4, text4, s5);
    } else {
      badge.className='badge err'; badge.textContent='OLLAMA ✗';
      box.className='ollama-status err';
      const s1 = document.createElement('span'); s1.style.color='var(--red)'; s1.textContent='● Offline';
      const s2 = document.createElement('span'); s2.style.color='var(--muted)';
      s2.appendChild(document.createTextNode('1. Install from ollama.com')); s2.appendChild(document.createElement('br'));
      s2.appendChild(document.createTextNode('2. ollama pull nomic-embed-text')); s2.appendChild(document.createElement('br'));
      s2.appendChild(document.createTextNode('3. ollama pull llama3.2'));
      const text1 = document.createTextNode('To enable RAG features:');
      box.append(s1, document.createElement('br'), document.createElement('br'), text1, document.createElement('br'), s2);
    }
  } catch(_) {}
}

async function insertDocument() {
  const title=document.getElementById('docTitle').value.trim();
  const text=document.getElementById('docText').value.trim();
  const btn=document.getElementById('insertDocBtn');
  const status=document.getElementById('insertStatus');
  if(!title||!text){status.textContent='⚠ Need both a title and text.';return;}

  btn.disabled=true; btn.textContent='Embedding…';
  status.replaceChildren();
  const sText = document.createElement('span'); sText.style.color = 'var(--muted)'; sText.textContent = 'Calling Ollama nomic-embed-text…';
  status.appendChild(sText);

  try {
    const r=await fetch(API+'/doc/insert',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({title,text})
    });
    const d=await r.json();
    if (d.error) {
      status.replaceChildren();
      const s = document.createElement('span'); s.style.color = 'var(--red)'; s.textContent = `✗ ${d.error}`;
      status.appendChild(s);
    } else {
      status.replaceChildren();
      const s = document.createElement('span'); s.style.color = 'var(--green)'; s.textContent = `✓ Inserted ${d.chunks} chunk(s) · ${d.dims}D embeddings`;
      status.appendChild(s);
      document.getElementById('docTitle').value='';
      document.getElementById('docText').value='';

          // Insert a 16D fake vector into the visualizer DB so it shows up on the map
          const emb16 = textToEmbedding(title + ' ' + text);
          fetch(API+'/insert', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body:JSON.stringify({metadata: title, category: 'doc', embedding: emb16})
          }).then(() => { loadItems().then(loadHNSW); });

      loadDocList(); checkOllamaStatus();
    }
  } catch(_) {
    status.replaceChildren();
    const s = document.createElement('span'); s.style.color = 'var(--red)'; s.textContent = '✗ Server error';
    status.appendChild(s);
  }
  btn.disabled=false; btn.textContent='⚡ EMBED & INSERT';
}

async function loadDocList() {
  try {
    const r=await fetch(API+'/doc/list'), docs=await r.json();
    document.getElementById('docCountLabel').textContent=docs.length;
    const container = document.getElementById('docList');
    container.replaceChildren();
    if (!docs.length) {
      const div = document.createElement('div'); div.style.color = 'var(--muted)'; div.style.fontSize = '11px'; div.textContent = 'No documents yet. Insert some above.';
      container.appendChild(div);
      return;
    }
    docs.forEach(d => {
      const dcard = document.createElement('div'); dcard.className = 'dcard';
      const title = document.createElement('div'); title.className = 'dcard-title'; title.textContent = d.title;
      const preview = document.createElement('div'); preview.className = 'dcard-preview'; preview.textContent = d.preview;
      const foot = document.createElement('div'); foot.className = 'dcard-foot';
      const words = document.createElement('span'); words.className = 'dcard-words'; words.textContent = `${d.words} words`;
      const delBtn = document.createElement('button'); delBtn.className = 'del'; delBtn.textContent = '✕';
      delBtn.onclick = () => deleteDoc(d.id);
      foot.append(words, delBtn);
      dcard.append(title, preview, foot);
      container.appendChild(dcard);
    });
  } catch(_) {}
}

async function deleteDoc(id) {
  try {
    await fetch(`${API}/doc/delete/${id}`,{method:'DELETE'});
    loadDocList(); checkOllamaStatus();
  } catch(_) {}
}

async function searchDocs() {
  const q = document.getElementById('docSearchInput').value.trim();
  if(!q) return;
  const container = document.getElementById('docSearchResults');
  
  const waitDiv = document.createElement('div');
  waitDiv.style.color = 'var(--muted)'; waitDiv.style.fontSize = '11px'; waitDiv.textContent = 'Searching...';
  container.replaceChildren(waitDiv);
  
  try {
    const r = await fetch(API+'/doc/search', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q, k: 5})
    });
    const d = await r.json();
    container.replaceChildren();
    if (d.error) {
      const err = document.createElement('div'); err.style.color = 'var(--red)'; err.style.fontSize = '11px'; err.textContent = 'Error: ' + d.error;
      container.appendChild(err);
    } else if (d.contexts && d.contexts.length) {
      d.contexts.forEach((ctx, i) => {
        const dcard = document.createElement('div'); dcard.className = 'dcard';
        const title = document.createElement('div'); title.className = 'dcard-title'; title.textContent = `#${i+1} ` + ctx.title;
        const foot = document.createElement('div'); foot.className = 'dcard-foot';
        const dist = document.createElement('span'); dist.className = 'dcard-words'; dist.textContent = `Distance: ${ctx.distance.toFixed(4)}`;
        foot.appendChild(dist);
        dcard.append(title, foot);
        container.appendChild(dcard);
      });
    } else {
      const none = document.createElement('div'); none.style.color = 'var(--muted)'; none.style.fontSize = '11px'; none.textContent = 'No matching documents found.';
      container.appendChild(none);
    }
  } catch(e) {
    container.replaceChildren();
    const err = document.createElement('div'); err.style.color = 'var(--red)'; err.style.fontSize = '11px'; err.textContent = 'Server error.';
    container.appendChild(err);
  }
}

document.getElementById('docSearchInput')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') searchDocs();
});

// ════════════════════════════════════════════════════════════
//  RAG  — Ask AI
// ════════════════════════════════════════════════════════════
document.getElementById('ragQuestion').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&e.ctrlKey)askAI();
});

async function askAI() {
  const question=document.getElementById('ragQuestion').value.trim();
  if(!question)return;
  const k=parseInt(document.getElementById('ragK').value);
  const btn=document.getElementById('askBtn');
  btn.disabled=true; btn.textContent='Thinking…';

  const history=document.getElementById('chatHistory');
  // Clear previous conversation
  history.replaceChildren();

  // Show question bubble
  const qDiv=document.createElement('div'); qDiv.className='chat-q';
  qDiv.textContent=question; history.appendChild(qDiv);

  // Show thinking indicator
  const thinkDiv=document.createElement('div'); thinkDiv.className='thinking';
  const spin = document.createElement('div'); spin.className = 'spinner';
  thinkDiv.append(spin, document.createTextNode('Retrieving context & generating answer…'));
  history.appendChild(thinkDiv);
  history.scrollTop=history.scrollHeight;

  // Update the scatter plot visualizer in the background using the real 768D DB
  fetch(API+'/doc/search', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, k})
  })
  .then(res=>res.json())
  .then(data => {
    if (data.contexts && data.contexts.length > 0) {
      hitIds = new Set();
      let sx=0, sy=0, sw=0;
      data.contexts.forEach((ctx, i) => {
        // Match the RAG chunk title back to the visualizer document metadata
        const pt = pcaPoints.find(p => p.item.category === 'doc' && ctx.title.startsWith(p.item.metadata));
        if (pt) {
          hitIds.add(pt.item.id);
          const w = 1/(i+1); sx += pt.x*w; sy += pt.y*w; sw += w;
        }
      });
      if (sw > 0) queryPt = {x: sx/sw + (Math.random()-.5)*.015, y: sy/sw + (Math.random()-.5)*.015};
    } else {
      hitIds = new Set();
      // Fallback: If no RAG docs match, visually move the star to the 16D semantic space anyway!
      const emb16 = textToEmbedding(question);
      fetch(`${API}/search?v=${emb16.join(',')}&k=3&metric=cosine&algo=hnsw`)
        .then(res2=>res2.json())
        .then(data2 => {
          if (data2.results && data2.results.length>0) {
            let sx=0,sy=0,sw=0;
            for (let i=0;i<Math.min(3,data2.results.length);i++) {
              const pt=pcaPoints.find(p=>p.item.id===data2.results[i].id);
              if(pt){const w=1/(i+1);sx+=pt.x*w;sy+=pt.y*w;sw+=w;}
            }
            if(sw>0) queryPt={x:sx/sw+(Math.random()-.5)*.015,y:sy/sw+(Math.random()-.5)*.015};
          }
        }).catch(()=>{});
    }
  }).catch(()=>{});

  try {
    const r=await fetch(API+'/doc/ask',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question,k})
    });
    const d=await r.json();
    thinkDiv.remove();

    const aDiv=document.createElement('div'); aDiv.className='chat-a';

    if (d.error) {
      const errLbl = document.createElement('div'); errLbl.className = 'chat-a-label'; errLbl.textContent = 'ERROR';
      const errTxt = document.createElement('div'); errTxt.className = 'chat-a-text'; errTxt.style.color = 'var(--red)'; errTxt.textContent = d.error;
      aDiv.append(errLbl, errTxt);
      history.appendChild(aDiv);
    } else {
      // Build answer block with typewriter
      const lbl = document.createElement('div'); lbl.className = 'chat-a-label'; lbl.textContent = `🤖 ${d.model||'llm'}`;
      const txt = document.createElement('div'); txt.className = 'chat-a-text'; txt.id = 'typeTarget';
      const ctx = document.createElement('div'); ctx.className = 'chat-ctx';
      const ctxLbl = document.createElement('div'); ctxLbl.className = 'chat-ctx-label'; ctxLbl.textContent = `RETRIEVED CONTEXT (${d.contexts.length} chunks)`;
      ctx.appendChild(ctxLbl);
      d.contexts.forEach((c,i)=>{
        const chip = document.createElement('span'); chip.className = 'ctx-chip';
        chip.onclick = () => toggleCtx(i);
        chip.textContent = `#${i+1} ${c.title} · ${c.distance.toFixed(3)}`;
        const exp = document.createElement('div'); exp.className = 'ctx-expand'; exp.id = `ctx-${i}`;
        exp.textContent = c.text;
        ctx.append(chip, exp);
      });
      aDiv.append(lbl, txt, ctx);
      history.appendChild(aDiv);

      // Typewriter effect
      const target=aDiv.querySelector('#typeTarget');
      target.classList.add('typing');
      const full=d.answer; let i=0;
      const timer=setInterval(()=>{
        if(i>=full.length){clearInterval(timer);target.classList.remove('typing');return;}
        const chunk=full.slice(i,i+3); target.textContent+=chunk; i+=3;
        history.scrollTop=history.scrollHeight;
      },18);
    }

  } catch(e) {
    thinkDiv.remove();
    const err=document.createElement('div');err.className='chat-a';
    const errLbl = document.createElement('div'); errLbl.className = 'chat-a-label'; errLbl.textContent = 'ERROR';
    const errTxt = document.createElement('div'); errTxt.className = 'chat-a-text'; errTxt.style.color = 'var(--red)'; errTxt.textContent = 'Server error — is the backend running?';
    err.append(errLbl, errTxt);
    history.appendChild(err);
  }

  document.getElementById('ragQuestion').value='';
  btn.disabled=false; btn.textContent='🤖 ASK AI';
  history.scrollTop=history.scrollHeight;
}

function toggleCtx(i) {
  const el=document.getElementById('ctx-'+i);
  el.style.display=el.style.display==='block'?'none':'block';
}

// ════════════════════════════════════════════════════════════
//  BOOT
// ════════════════════════════════════════════════════════════
resize(); drawFrame();
loadItems().then(loadHNSW);
checkOllamaStatus();