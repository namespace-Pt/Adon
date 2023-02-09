### default
torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=1 ++y_query_gate_k=1
torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=1
torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=2
torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=3
torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=4
torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=5
torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=8 ++x_posting_prune=0.999

0.3843|0.6489|0.86|0.9096|0.06|11152|
0.3924|0.6669|0.8885|0.9445|0.14|26018|
0.3977|0.6783|0.9078|0.9664|0.2|36605|
0.3982|0.6808|0.9127|0.9723|0.28|51216|
0.3989|0.6833|0.9167|0.9772|0.39|71286|
0.3991|0.6838|0.9185|0.9799|0.53|98207|
0.3992|0.6844|0.9207|0.9831|1.22|224576|


### IVFOPQ
# python run.py RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=10 ++load_index ++load_encode
# python run.py RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=50 ++load_index ++load_encode
# python run.py RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=100 ++load_index ++load_encode
# python run.py RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=200 ++load_index ++load_encode
# python run.py RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=500 ++load_index ++load_encode
# python run.py RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=750 ++load_index ++load_encode
# python run.py RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=1000 ++load_index ++load_encode

# 0.2631|0.4452|0.606|0.6502|15432|
# 0.3285|0.5558|0.7509|0.8047|71490|
# 0.3458|0.5903|0.7958|0.8533|136981|
# 0.3562|0.6143|0.8293|0.8887|260741|
# 0.3691|0.641|0.8673|0.929|605036|
# 0.3738|0.6501|0.8797|0.9424|876616|
# 0.3778|0.6566|0.8882|0.9516|1139294|


### DistillVQ
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=10 ++load_index ++load_encode ++mode=eval ++load_ckpt=none
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=20 ++load_index ++load_encode ++mode=eval ++load_ckpt=none
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=50 ++load_index ++load_encode ++mode=eval ++load_ckpt=none
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=100 ++load_index ++load_encode ++mode=eval ++load_ckpt=none
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=200 ++load_index ++load_encode ++mode=eval ++load_ckpt=none
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=500 ++load_index ++load_encode ++mode=eval ++load_ckpt=none
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=750 ++load_index ++load_encode ++mode=eval ++load_ckpt=none
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=1000 ++load_index ++load_encode ++mode=eval ++load_ckpt=none
python run.py DistillVQ_d-RetroMAE ++index_type=\'OPQ96,IVF10000,PQ96x8\' ++nprobe=1200 ++load_index ++load_encode ++mode=eval ++load_ckpt=none

0.3242|0.5476|0.7362|0.7905|9737|
0.3394|0.5774|0.7805|0.8397|19173|
0.3518|0.6036|0.8213|0.8881|46869|
0.3583|0.6177|0.8427|0.9138|91924|
0.363|0.627|0.8572|0.9316|179999|
0.3681|0.6375|0.8738|0.9512|436620|
0.37|0.6415|0.8801|0.9585|650163|
0.3705|0.6427|0.8819|0.9607|864030|
0.3709|0.6438|0.8836|0.9627|1034947|



### w.o. Terms
# torchrun --nproc_per_node=4 run.py TopIVF ++mode=eval ++query_gate_k=10
# torchrun --nproc_per_node=4 run.py TopIVF ++mode=eval ++query_gate_k=25
# torchrun --nproc_per_node=4 run.py TopIVF ++mode=eval ++query_gate_k=50
# torchrun --nproc_per_node=4 run.py TopIVF ++mode=eval ++query_gate_k=75
# torchrun --nproc_per_node=4 run.py TopIVF ++mode=eval ++query_gate_k=100
# torchrun --nproc_per_node=4 run.py TopIVF ++mode=eval ++query_gate_k=150
# torchrun --nproc_per_node=4 run.py TopIVF ++mode=eval ++query_gate_k=200

# 0.334|0.5583|0.739|0.7852|10027|
# 0.359|0.6034|0.8009|0.8537|24798|
# 0.3693|0.6254|0.8337|0.8892|49191|
# 0.3755|0.636|0.8489|0.9058|73516|
# 0.378|0.641|0.8562|0.9143|97837|
# 0.3821|0.6483|0.8679|0.9275|146463|
# 0.3848|0.6532|0.8744|0.9348|195309|


### w.o. Topics
torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++text_gate_k=1 ++load_encode
torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++text_gate_k=2 ++load_encode
torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++text_gate_k=3 ++load_encode
torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++text_gate_k=4 ++load_encode
torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++text_gate_k=5 ++load_encode
torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++text_gate_k=8 ++load_encode
torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++text_gate_k=10 ++posting_prune=0.999 ++load_encode

0.3539|0.5835|0.7409|0.7639|6118|
0.3951|0.6717|0.8854|0.9301|31316|
0.397|0.6782|0.8999|0.9505|51386|
0.3977|0.6804|0.9053|0.9607|78307|
0.399|0.684|0.9152|0.974|204676|
0.3989|0.6845|0.9166|0.9763|203328|



### without distill
# torchrun --nproc_per_node=4 run.py TokIVF ++mode=eval ++load_ckpt=UniCOIL_title/backup ++save_encode++save_ckpt=contra ++save_model
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=1 ++y_query_gate_k=5 ++verifier_src=RetroMAE ++x_load_ckpt=contra
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=1 ++verifier_src=RetroMAE ++x_load_ckpt=contra
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=2 ++verifier_src=RetroMAE ++x_load_ckpt=contra
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=3 ++verifier_src=RetroMAE ++x_load_ckpt=contra
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=5 ++verifier_src=RetroMAE ++x_posting_prune=0.999 ++x_load_ckpt=contra
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=6 ++verifier_src=RetroMAE ++x_posting_prune=0.999 ++x_load_ckpt=contra
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_text_gate_k=8 ++x_posting_prune=0.999 ++verifier_src=RetroMAE ++x_load_ckpt=contra

# 0.3677|0.6347|0.8461|0.8991|0.07|12132|
# 0.3758|0.6532|0.8762|0.938|0.15|26998|
# 0.3811|0.6651|0.8983|0.963|0.23|42598|
# 0.382|0.6688|0.9055|0.9705|0.37|68389|
# 0.3829|0.6704|0.91|0.9767|0.61|112230|
# 0.383|0.671|0.9117|0.9787|0.76|139119|
# 0.383|0.671|0.9131|0.9816|1.05|193934|


### without train
# torchrun --nproc_per_node=8 run.py BM25 ++pretokenize ++text_col=[1,2] ++save_encode ++mode=encode
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=1 ++y_query_gate_k=3 ++verifier_src=RetroMAE ++x_pretokenize
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=1 ++verifier_src=RetroMAE ++x_pretokenize
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=3 ++verifier_src=RetroMAE ++x_pretokenize
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=5 ++verifier_src=RetroMAE ++x_pretokenize
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=8 ++verifier_src=RetroMAE ++x_pretokenize
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=11 ++verifier_src=RetroMAE ++x_pretokenize
# torchrun --nproc_per_node=4 run.py BIVFPQ ++x_model=BM25 ++x_load_ckpt=inv ++y_model=IVF ++x_text_gate_k=16 ++verifier_src=RetroMAE ++x_pretokenize


# 0.3378|0.5787|0.7652|0.8106|0.07|12688|
# 0.3547|0.6123|0.8293|0.8741|0.19|35233|
# 0.3736|0.6484|0.874|0.9344|0.28|50924|
# 0.3775|0.6608|0.8908|0.954|0.38|69764|
# 0.381|0.6664|0.9017|0.9667|0.54|99150|
# 0.3822|0.6689|0.9065|0.9731|0.71|131655|
# 0.383|0.6716|0.9112|0.9786|1.15|212753|


# torchrun --nproc_per_node=4 --master_port=12345 run.py BIVFPQ ++x_posting_prune=0.5
# torchrun --nproc_per_node=4 --master_port=12345 run.py BIVFPQ ++x_posting_prune=0.7
# torchrun --nproc_per_node=4 --master_port=12345 run.py BIVFPQ ++x_posting_prune=0.9
# torchrun --nproc_per_node=4 --master_port=12345 run.py BIVFPQ ++x_posting_prune=0.99
# torchrun --nproc_per_node=4 --master_port=12345 run.py BIVFPQ ++x_posting_prune=0.996
# torchrun --nproc_per_node=4 --master_port=12345 run.py BIVFPQ ++x_posting_prune=0.999
# torchrun --nproc_per_node=4 --master_port=12345 run.py BIVFPQ ++x_posting_prune=0

# 0.3659|0.6183|0.821|0.8728|0.12|21915|
# 0.377|0.6406|0.8498|0.9035|0.13|24083|
# 0.3928|0.6703|0.8907|0.9464|0.17|31151|
# 0.3978|0.6812|0.9094|0.9694|0.23|43201|
# 0.398|0.6817|0.9103|0.9709|0.25|45683|
# 0.398|0.6819|0.9112|0.9723|0.26|48209|
# 0.3979|0.6818|0.9113|0.9726|0.28|51216|
