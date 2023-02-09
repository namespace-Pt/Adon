### default
torchrun --nproc_per_node=4 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=1 ++y_query_gate_k=2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=1
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=3
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=4
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=5
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=8

# 0.7565|0.8036|0.8332|0.8742|0.06|28355|
# 0.7698|0.8205|0.8488|0.8909|0.13|55693|
# 0.7778|0.8285|0.8582|0.9003|0.21|90197|
# 0.7795|0.8319|0.8607|0.9042|0.3|132415|
# 0.7798|0.833|0.8626|0.9064|0.41|178768|
# 0.7809|0.8335|0.8637|0.9078|0.53|230080|
# 0.7806|0.8341|0.8643|0.9086|0.89|389082|


### IVFOPQ
# python run.py AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=10 ++load_index ++load_encode ++device=1 ++load_ckpt=none
python run.py AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=30 ++load_index ++load_encode ++device=1 ++load_ckpt=none
# python run.py AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=50 ++load_index ++load_encode ++device=1 ++load_ckpt=none
# python run.py AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=100 ++load_index ++load_encode ++device=1 ++load_ckpt=none
# python run.py AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=200 ++load_index ++load_encode ++device=1 ++load_ckpt=none
# python run.py AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=500 ++load_index ++load_encode ++device=1 ++load_ckpt=none
# python run.py AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=750 ++load_index ++load_encode ++device=1 ++load_ckpt=none
# python run.py AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=1000 ++load_index ++load_encode ++device=1 ++load_ckpt=none

# 0.544|0.5953|0.6388|0.715|30937|
# 0.6548|0.7058|0.7396|0.8047|155988|
# 0.6864|0.7424|0.7748|0.8363|311502|
# 0.7161|0.769|0.8025|0.8568|620229|
# 0.741|0.7939|0.8277|0.8792|1529677|
# 0.7587|0.8125|0.8476|0.8939|2983870|


### DistillVQ
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=10 ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2 ++load_ckpt=ivfopq96 ++save_index
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=20 ++load_index ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=50 ++load_index ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=100 ++load_index ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=200 ++load_index ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=500 ++load_index ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=750 ++load_index ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=1000 ++load_index ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2
# python run.py DistillVQ_d-AR2 base=NQ-open ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=1200 ++load_index ++load_encode ++mode=eval ++device=1 ++embedding_src=AR2 ++vq_src=AR2

0.6618|0.7163|0.7529|0.8127|18587|
0.682|0.7357|0.7726|0.8327|36985|
0.6964|0.751|0.7886|0.8488|87932|
0.7053|0.7615|0.7992|0.8596|160504|
0.713|0.7709|0.8069|0.8687|308437|
0.718|0.7767|0.8139|0.8748|849536|
0.7197|0.7781|0.8155|0.8767|1332890|
0.7205|0.7784|0.8161|0.8773|1825239|
0.7208|0.7787|0.8163|0.8776|2227121|
0.7587|0.8125|0.8476|0.8939|2983870|


### w.o. Terms
# torchrun --nproc_per_node=4 --master_port=12345 run.py TopIVF base=NQ-open ++mode=eval ++query_gate_k=10 ++embedding_src=AR2 ++verifier_src=DistillVQ_d-AR2 ++vq_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py TopIVF base=NQ-open ++mode=eval ++query_gate_k=25 ++embedding_src=AR2 ++verifier_src=DistillVQ_d-AR2 ++vq_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py TopIVF base=NQ-open ++mode=eval ++query_gate_k=50 ++embedding_src=AR2 ++verifier_src=DistillVQ_d-AR2 ++vq_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py TopIVF base=NQ-open ++mode=eval ++query_gate_k=75 ++embedding_src=AR2 ++verifier_src=DistillVQ_d-AR2 ++vq_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py TopIVF base=NQ-open ++mode=eval ++query_gate_k=100 ++embedding_src=AR2 ++verifier_src=DistillVQ_d-AR2 ++vq_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py TopIVF base=NQ-open ++mode=eval ++query_gate_k=150 ++embedding_src=AR2 ++verifier_src=DistillVQ_d-AR2 ++vq_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py TopIVF base=NQ-open ++mode=eval ++query_gate_k=200 ++embedding_src=AR2 ++verifier_src=DistillVQ_d-AR2 ++vq_src=AR2

# 0.7111|0.7532|0.785|0.833|17487|
# 0.7416|0.7875|0.8177|0.8659|44935|
# 0.7546|0.8039|0.8335|0.8812|89768|
# 0.7587|0.8108|0.8388|0.8853|135634|
# 0.7645|0.8161|0.8432|0.8903|182055|
# 0.7698|0.8211|0.8493|0.8947|277290|
# 0.7723|0.8258|0.8532|0.8986|375286|


### w.o. Topics
# torchrun --nproc_per_node=2 --master_port=12345 run.py TokIVF base=NQ-open ++mode=eval ++text_gate_k=1 ++load_encode ++verifier_src=DistillVQ_d-AR2 ++posting_prune=0.996
# torchrun --nproc_per_node=2 --master_port=12345 run.py TokIVF base=NQ-open ++mode=eval ++text_gate_k=2 ++load_encode ++verifier_src=DistillVQ_d-AR2 ++posting_prune=0.996
# torchrun --nproc_per_node=2 --master_port=12345 run.py TokIVF base=NQ-open ++mode=eval ++text_gate_k=3 ++load_encode ++verifier_src=DistillVQ_d-AR2 ++posting_prune=0.996
# torchrun --nproc_per_node=2 --master_port=12345 run.py TokIVF base=NQ-open ++mode=eval ++text_gate_k=4 ++load_encode ++verifier_src=DistillVQ_d-AR2 ++posting_prune=0.996
# torchrun --nproc_per_node=2 --master_port=12345 run.py TokIVF base=NQ-open ++mode=eval ++text_gate_k=5 ++load_encode ++verifier_src=DistillVQ_d-AR2 ++posting_prune=0.996
# torchrun --nproc_per_node=2 --master_port=12345 run.py TokIVF base=NQ-open ++mode=eval ++text_gate_k=8 ++load_encode ++verifier_src=DistillVQ_d-AR2 ++posting_prune=0.996
# torchrun --nproc_per_node=2 --master_port=12345 run.py TokIVF base=NQ-open ++mode=eval ++text_gate_k=10 ++load_encode ++verifier_src=DistillVQ_d-AR2 ++posting_prune=0.996

# 0.682|0.7258|0.7565|0.805|19847|
# 0.7488|0.7947|0.8235|0.8648|54351|
# 0.7657|0.8144|0.8429|0.8867|96569|
# 0.7715|0.8213|0.851|0.8947|142922|
# 0.7765|0.8271|0.8582|0.8986|194234|
# 0.7809|0.833|0.862|0.9055|353236|
# 0.7825|0.833|0.8623|0.9064|455751|


### without distill
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=1 ++y_query_gate_k=5 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=1 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=2 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=3 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=4 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=5 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_text_gate_k=8 ++verifier_src=AR2

# 0.7468|0.7983|0.8321|0.8745|0.06|28355|
# 0.7598|0.8139|0.8488|0.8906|0.13|55693|
# 0.7698|0.823|0.8565|0.8986|0.21|90197|
# 0.7701|0.8241|0.8582|0.9025|0.3|132415|
# 0.7706|0.826|0.8607|0.9044|0.41|178768|
# 0.772|0.8274|0.8615|0.9055|0.53|230080|
# 0.7723|0.8271|0.8626|0.9064|0.89|389082|


### without train
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=1 ++y_query_gate_k=5 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=3 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=5 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=8 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=11 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=14 ++verifier_src=AR2
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=20 ++verifier_src=AR2

# 0.6886|0.7402|0.7748|0.823|0.05|23633|
# 0.7485|0.7997|0.833|0.8806|0.23|101344|
# 0.7557|0.808|0.8418|0.8875|0.32|141657|
# 0.7598|0.8147|0.8485|0.8928|0.46|200877|
# 0.7634|0.818|0.8512|0.8956|0.58|251809|
# 0.767|0.8222|0.8551|0.8981|0.68|296296|
# 0.7695|0.8252|0.8593|0.9006|0.87|381999|



### posting prune
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_posting_prune=0.5
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_posting_prune=0.7
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_posting_prune=0.9
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_posting_prune=0.99
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_posting_prune=0.996
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_posting_prune=0.999
# torchrun --nproc_per_node=2 --master_port=12345 run.py BIVFPQ-NQ ++x_posting_prune=0

# 0.7609|0.8089|0.8385|0.8837|0.1|43021|
# 0.7668|0.8163|0.8468|0.8898|0.11|48806|
# 0.7734|0.8249|0.8551|0.8978|0.14|63210|
# 0.7787|0.8316|0.8609|0.9039|0.25|108008|
# 0.7795|0.8319|0.8607|0.9042|0.3|132415|
# 0.7792|0.8321|0.8609|0.9044|0.42|181919|
# 0.7792|0.8321|0.8615|0.9053|15.81|6920138|

torchrun --nproc_per_node=4 run.py BIVFPQ-NQ ++y_model=TopIVF_RetroMAE ++verifier_src=DistillVQ_d-RetroMAE
