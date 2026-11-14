import test from "node:test";
import assert from "node:assert/strict";
import {parseLearningRoute} from "../src/learning-view.mjs";

test("accept canonical deep links and percent-encoded topic IDs",()=>{
 assert.deepEqual(parseLearningRoute("#station/1"),{kind:"station",order:1});
 assert.deepEqual(parseLearningRoute("#station/44"),{kind:"station",order:44});
 assert.deepEqual(parseLearningRoute("#chapter/9"),{kind:"chapter",index:9});
 assert.deepEqual(parseLearningRoute("#topic/NLP-003"),{kind:"topic",id:"NLP-003"});
 assert.deepEqual(parseLearningRoute("#topic/NLP%2D003"),{kind:"topic",id:"NLP-003"});
});
test("reject malformed URL escapes and invalid or ambiguous routes",()=>{
 for(const hash of [
  "#topic/%", "#topic/%ZZ", "#topic/%E0%A4%A", "#station/%FF",
  "#station/0", "#station/45", "#station/01", "#chapter/0",
  "#chapter/10", "#topic/", "#topic/NLP-003/extra", "#station/1?x=1",
  "#topic/javascript:alert(1)", "#topic/%2e%2e%2fsecret",
  "#topic/NLP-003%00", "", null, undefined
 ])assert.equal(parseLearningRoute(hash),null,String(hash));
});
test("never interpret a catalog route as a URL or external redirect",()=>{
 for(const hash of ["#https://example.com", "#//example.com", "#topic/%2F%2Fevil.example"]){
  assert.equal(parseLearningRoute(hash),null,hash);
 }
});
