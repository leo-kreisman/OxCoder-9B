// Independent llama.cpp oracle, used only by tests, never by Kindi inference.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
int main(int argc,char**argv){
 if(argc!=5)return 2;
 std::ifstream input(argv[2]);std::string prompt((std::istreambuf_iterator<char>(input)),{});
 ggml_backend_load_all();llama_backend_init();
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;
 auto model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=128;cp.n_ubatch=128;cp.n_threads=2;cp.n_threads_batch=2;
 auto ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
 auto vocab=llama_model_get_vocab(model);const int V=llama_vocab_n_tokens(vocab);
 std::vector<llama_token> tokens(prompt.size()+64);
 int n=llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),tokens.size(),false,true);
 if(n<=0||n>=128)return 5;tokens.resize(n);
 std::string prefix=argv[3];std::ofstream ids(prefix+".prompt-ids");for(auto t:tokens)ids<<t<<"\n";ids.close();
 if(llama_decode(ctx,llama_batch_get_one(tokens.data(),tokens.size())))return 6;
 int count=std::stoi(argv[4]);std::ofstream out(prefix+".ids");
 for(int i=0;i<count;i++){
  auto logits=llama_get_logits_ith(ctx,-1);
  if(i==0){std::ofstream f(prefix+".logits",std::ios::binary);f.write((char*)logits,V*sizeof(float));}
  llama_token best=std::max_element(logits,logits+V)-logits;out<<best<<"\n";out.flush();
  char b[512];int nb=llama_token_to_piece(vocab,best,b,sizeof(b),0,true);if(nb>0)fwrite(b,1,nb,stdout);fflush(stdout);
  if(i+1<count&&llama_decode(ctx,llama_batch_get_one(&best,1)))return 7;
 }
 llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
