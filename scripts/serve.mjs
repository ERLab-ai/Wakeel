/**
 * Zero-dependency, Windows-compatible development preview of the six-file
 * allowlisted /dist product. This is NOT a production server.
 */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root=path.resolve(process.cwd(),"dist");
const whitelist=new Set([
 "/","/index.html","/styles.css","/src/app.mjs",
 "/src/models.mjs","/src/learning-view.mjs","/data/learning.json"
]);
const types={".html":"text/html; charset=utf-8",".css":"text/css; charset=utf-8",
 ".mjs":"text/javascript; charset=utf-8",".json":"application/json; charset=utf-8"};

export function handler(req,res){
 let url;
 try{url=new URL(req.url,"http://localhost");}
 catch{res.writeHead(400);res.end("Bad request");return;}
 const pathname=url.pathname==="/" ? "/index.html" : url.pathname;
 if(!whitelist.has(pathname)){res.writeHead(404,{"Content-Type":"text/plain; charset=utf-8"});res.end("Not found");return;}
 const file=path.join(root,...pathname.slice(1).split("/"));
 readFile(file).then(content=>{
  res.writeHead(200,{"Content-Type":types[path.extname(file)]||"application/octet-stream",
    "Cache-Control":"no-store","X-Content-Type-Options":"nosniff"});
  res.end(content);
 }).catch(()=>{res.writeHead(404);res.end("Not found")});
}

if(process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const port=Number(process.env.PORT||8080);
  createServer(handler).listen(port,"127.0.0.1",()=>console.log("Wakeel preview: http://127.0.0.1:"+port));
}