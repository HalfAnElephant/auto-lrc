/* All annotation coordinates refer to the source PCM sample grid. */
let state, rowIndex = 0, segmentIndex = 0, start = 0, end = 0, selected = null, signal = null, requestId = 0, dirty = false;
const $ = id => document.getElementById(id);
const audio = $('audio');
const names = {pending:'待复核', reviewed:'已标注', ambiguous:'边界不明确'};
const icons = () => lucide.createIcons();
const current = () => state.rows[rowIndex];
const metadata = () => state.segments[segmentIndex];
const message = text => { $('message').textContent = text; };
async function api(path, options) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}
function redraw() {
  if (!signal) return;
  for (const [id, height] of [['waveform',160],['spectrum',190]]) {
    const canvas = $(id), ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, h = canvas.clientHeight || height;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(h * ratio);
    const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio); ctx.clearRect(0,0,width,h);
    if (id === 'waveform') {
      ctx.strokeStyle='#e0e7e8'; ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(width,h/2);ctx.stroke();
      ctx.strokeStyle='#257d70';ctx.beginPath();
      signal.waveform.forEach((point,i) => {const x=(i+.5)/signal.waveform.length*width;ctx.moveTo(x,h/2-point[0]*h*.46);ctx.lineTo(x,h/2-point[1]*h*.46);});ctx.stroke();
    } else {
      ctx.fillStyle='#111b1f';ctx.fillRect(0,0,width,h);
      const columns=signal.spectrogram_db, frequencies=signal.frequencies_hz;
      const hop=columns.length>1?signal.spectrogram_centers_sample[1]-signal.spectrogram_centers_sample[0]:end-start;
      const maximumFrequency=frequencies[frequencies.length-1] || 1;
      columns.forEach((column,i) => column.forEach((db,j) => {
        const value=Math.max(0,Math.min(1,(db+100)/85));
        const red=Math.round(15+value*215),green=Math.round(24+Math.sin(value*Math.PI*.75)*175),blue=Math.round(35+(1-value)*60);
        ctx.fillStyle=`rgb(${red},${green},${blue})`;
        const x=(signal.spectrogram_centers_sample[i]-hop/2-start)/(end-start)*width;
        const y=h-frequencies[j]/maximumFrequency*h;
        const rowHeight=j?Math.max(1,(frequencies[j]-frequencies[j-1])/maximumFrequency*h):h/frequencies.length;
        ctx.fillRect(x,y,Math.max(1,hop/(end-start)*width+1),rowHeight+1);
      }));
    }
    if (selected !== null && selected>=start && selected<end) {
      const x=(selected-start)/(end-start)*width;
      ctx.strokeStyle=id==='waveform'?'#c64155':'#ff6d87';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
    }
  }
}
async function loadSignal() {
  const id=++requestId;
  try {
    const result=await api(`/api/signal?segment=${segmentIndex}&start=${start}&end=${end}&width=${Math.max(64,Math.min(2000,Math.round($('waveform').clientWidth)))}`);
    if (id!==requestId) return;
    signal=result;redraw();
    $('window').textContent=`${(start/metadata().sample_rate).toFixed(3)}–${(end/metadata().sample_rate).toFixed(3)} s`;
    $('ticks').replaceChildren(...Array.from({length:5},(_,i)=>{const el=document.createElement('span');el.textContent=((start+(end-start)*i/4)/metadata().sample_rate).toFixed(3)+' s';return el;}));
  } catch(error) {message(error.message);}
}
function setSample(value, edit=true) {
  selected=value===null?null:Math.max(0,Math.min(metadata().num_samples-1,Math.round(value)));
  $('sample').value=selected===null?'':selected;
  $('seconds').textContent=selected===null?'未选择':(selected/metadata().sample_rate).toFixed(6);
  if(edit) dirty=true;redraw();
}
function unitButtons() {
  $('units').replaceChildren();
  state.rows.forEach((row,index)=>{
    if(row.segment_id!==metadata().segment_id)return;
    const li=document.createElement('li'),button=document.createElement('button');
    button.className=index===rowIndex?'active':'';
    const number=document.createElement('span'),text=document.createElement('span'),mark=document.createElement('span');
    number.textContent=row.unit_index+1;text.textContent=row.text;mark.className='mark';mark.textContent=row.review_status==='pending'?'':row.review_status==='ambiguous'?'?':'✓';
    button.append(number,text,mark);button.onclick=()=>choose(index);li.append(button);$('units').append(li);
  });
  $('progress').textContent=`${state.rows.filter(r=>r.review_status!=='pending').length} / ${state.rows.length}`;
}
function choose(index) {
  if(dirty && !window.confirm('当前标注尚未保存，继续切换？'))return;
  audio.pause();dirty=false;rowIndex=index;
  const row=current(),newSegment=state.segments.findIndex(s=>s.segment_id===row.segment_id);
  if(newSegment!==segmentIndex || !signal){
    segmentIndex=newSegment;start=0;end=metadata().num_samples;signal=null;audio.src=`/audio/${segmentIndex}`;loadSignal();
  }
  $('segment').value=segmentIndex;
  $('metadata').textContent=`${metadata().sample_rate.toLocaleString()} Hz · ${(metadata().num_samples/metadata().sample_rate).toFixed(3)} s · ${metadata().channels} ch`;
  $('position').textContent=`${row.language} / ${row.unit_index+1}`;
  $('word').textContent=row.text;$('status').textContent=names[row.review_status];
  $('reviewer').value=row.reviewer || localStorage.getItem('boundary-reviewer') || '';
  $('confidence').value=row.confidence ?? .8;$('confidence-value').textContent=$('confidence').value;
  $('ambiguous').checked=row.review_status==='ambiguous';$('notes').value=row.review_notes || '';
  $('sample').disabled=$('ambiguous').checked;setSample(row.human_truth_sample,false);unitButtons();message('');
}
function zoom(factor,pan=0) {
  const duration=metadata().num_samples,span=Math.min(duration,Math.max(96,Math.round((end-start)*factor)));
  const center=(selected!==null && selected>=start && selected<end?selected:(start+end)/2)+pan*(end-start);
  start=Math.max(0,Math.min(duration-span,Math.round(center-span/2)));end=start+span;loadSignal();
}
for(const id of ['waveform','spectrum'])$(id).onclick=event=>{
  if($('ambiguous').checked)return;
  const rectangle=event.currentTarget.getBoundingClientRect();setSample(start+(event.clientX-rectangle.left)/rectangle.width*(end-start));
};
$('sample').oninput=()=>setSample($('sample').value===''?null:Number($('sample').value));
$('previous-sample').onclick=()=>{if(selected!==null && !$('ambiguous').checked)setSample(selected-1);};
$('next-sample').onclick=()=>{if(selected!==null && !$('ambiguous').checked)setSample(selected+1);};
$('confidence').oninput=()=>{$('confidence-value').textContent=$('confidence').value;dirty=true;};
for(const id of ['reviewer','notes'])$(id).oninput=()=>{dirty=true;};
$('ambiguous').onchange=()=>{$('sample').disabled=$('ambiguous').checked;if($('ambiguous').checked)setSample(null);dirty=true;};
$('zoom-in').onclick=()=>zoom(.5);$('zoom-out').onclick=()=>zoom(2);
$('pan-left').onclick=()=>zoom(1,-.75);$('pan-right').onclick=()=>zoom(1,.75);
$('reset-view').onclick=()=>{start=0;end=metadata().num_samples;loadSignal();};
$('segment').onchange=()=>{const index=Number($('segment').value);choose(state.rows.findIndex(r=>r.segment_id===state.segments[index].segment_id));$('segment').value=segmentIndex;};
$('next').onclick=()=>choose((rowIndex+1)%state.rows.length);
let playTimer;
$('play').onclick=async()=>{clearInterval(playTimer);audio.currentTime=start/metadata().sample_rate;audio.playbackRate=Number($('speed').value);try{await audio.play();playTimer=setInterval(()=>{if(audio.currentTime>=end/metadata().sample_rate){audio.pause();clearInterval(playTimer);}},15);}catch(error){message(error.message);}};
$('stop').onclick=()=>{audio.pause();clearInterval(playTimer);};
$('speed').onchange=()=>{audio.playbackRate=Number($('speed').value);};
$('save').onclick=async()=>{
  $('save').disabled=true;
  try{
    const ambiguous=$('ambiguous').checked;
    if(!ambiguous && selected===null)throw new Error('请选择起点采样点');
    const reviewer=$('reviewer').value.trim();
    const label={human_truth_sample:ambiguous?null:selected,human_truth_seconds:null,
      confidence:Number($('confidence').value),review_status:ambiguous?'ambiguous':'reviewed',ambiguous,
      reviewer,review_notes:$('notes').value || null};
    const result=await api('/api/label',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':state.token},body:JSON.stringify({row_index:rowIndex,label,revision:state.revision})});
    state.rows=result.rows;state.revision=result.revision;dirty=false;localStorage.setItem('boundary-reviewer',reviewer);
    $('status').textContent=names[current().review_status];unitButtons();message('已保存');
  }catch(error){message(error.message);}finally{$('save').disabled=false;}
};
window.addEventListener('resize',redraw);
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
async function init(){
  icons();state=await api('/api/form');
  state.segments.forEach((segment,i)=>{const option=document.createElement('option');option.value=i;const parts=segment.segment_id.split('/');option.textContent=`${parts[3]} · ${parts[1]} · ${parts[4]} · ${parts[5]}`;$('segment').append(option);});
  choose(state.rows.findIndex(r=>r.review_status==='pending')<0?0:state.rows.findIndex(r=>r.review_status==='pending'));
}
init().catch(error=>message(error.message));
