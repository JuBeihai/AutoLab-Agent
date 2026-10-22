const fs = require('fs'), vm = require('vm'), assert = require('assert');
class Element {
  constructor(tag) {this.tag=tag;this.children=[];this.textContent='';}
  appendChild(x) {this.children.push(x);return x;}
}
const source=fs.readFileSync(require('path').join(__dirname,'../webui/static/app.js'),'utf8');
const linkFunction=source.slice(source.indexOf('function codeDownloadLink('),source.indexOf("$('#btn-code-workspace')"));
const renderFunction=source.slice(source.indexOf('function appendCodeResults('),source.indexOf('function appendWebSources('));
const context=vm.createContext({document:{createElement:tag=>new Element(tag)}});
vm.runInContext(linkFunction+renderFunction,context);
const bubble=new Element('div');
context.appendCodeResults({bubble},{workspace:'/work/S1',generated_files:[
  {path:'<script>alert(1)</script>.py',download_url:'/api/code-workspace/S1/download?path=files%2Fx.py'},
  {path:'bad',download_url:'javascript:alert(1)'}],code_runs:[{
    path:'test.py',status:'failed',exit_code:1,seconds:.2,stderr:'<script>raw traceback</script>',
    files:[{path:'stderr.txt',download_url:'/api/code-workspace/S1/download?path=runs%2Fx%2Fstderr.txt'}]}]});
const all=[];function walk(e){all.push(e);e.children.forEach(walk);}walk(bubble);
assert.equal(all.filter(e=>e.tag==='a').length,2);
assert(all.some(e=>e.tag==='pre'&&e.textContent.includes('<script>raw traceback</script>')));
assert(all.some(e=>e.textContent.includes('执行失败')));
assert(all.every(e=>!Object.hasOwn(e,'innerHTML')));
console.log('Code result rendering, download links, failure display and inert output passed');
