import pandas as pd


all_doc_count = {
    "marco": 8848123,
    "nq": 21015324
}

ivfopq_marco = """
0.2631|0.4452|0.606|0.6502|15432|
0.3285|0.5558|0.7509|0.8047|71490|
0.3458|0.5903|0.7958|0.8533|136981|
0.3562|0.6143|0.8293|0.8887|260741|
0.3691|0.641|0.8673|0.929|605036|
0.3738|0.6501|0.8797|0.9424|876616|
0.3778|0.6566|0.8882|0.9516|1139294|
"""
data_ivfopq_marco = []
for i,line in enumerate(ivfopq_marco.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[2])
    posting_length = int(fields[-2])
    data_ivfopq_marco.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "IVFOPQ",
        "portion": posting_length / all_doc_count["marco"],
        "inverse-portion": round(all_doc_count["marco"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["marco"]*48),
    })
data_ivfopq_marco = pd.DataFrame(data_ivfopq_marco)


distillvq_marco = """
0.3242|0.5476|0.7362|0.7905|9737|
0.3394|0.5774|0.7805|0.8397|19173|
0.3518|0.6036|0.8213|0.8881|46869|
0.3583|0.6177|0.8427|0.9138|91924|
0.363|0.627|0.8572|0.9316|179999|
0.3681|0.6375|0.8738|0.9512|436620|
0.37|0.6415|0.8801|0.9585|650163|
"""
data_distillvq_marco = []
for i,line in enumerate(distillvq_marco.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[2])
    posting_length = int(fields[-2])
    data_distillvq_marco.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "Distill-VQ",
        "portion": posting_length / all_doc_count["marco"],
        "inverse-portion": round(all_doc_count["marco"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["marco"]*48),
    })
data_distillvq_marco = pd.DataFrame(data_distillvq_marco)


bivfpq_marco = """
0.3804|0.6416|0.8465|0.8935|0.05|9145|
0.3924|0.6669|0.8885|0.9445|0.14|26018|
0.3977|0.6783|0.9078|0.9664|0.2|36605|
0.3982|0.6808|0.9127|0.9723|0.28|51216|
0.3989|0.6833|0.9167|0.9772|0.39|71286|
0.3991|0.6838|0.9185|0.9799|0.53|98207|
0.3992|0.6844|0.9207|0.9831|1.22|224576|
"""
data_bivfpq_marco = []
for i,line in enumerate(bivfpq_marco.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[2])
    posting_length = int(fields[5])
    data_bivfpq_marco.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "Bi-IVFPQ",
        "portion": posting_length / all_doc_count["marco"],
        "inverse-portion": round(all_doc_count["marco"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["marco"]*48),
    })
data_bivfpq_marco = pd.DataFrame(data_bivfpq_marco)


noterms_marco = """
0.334|0.5583|0.739|0.7852|10027|
0.359|0.6034|0.8009|0.8537|24798|
0.3693|0.6254|0.8337|0.8892|49191|
0.3755|0.636|0.8489|0.9058|73516|
0.378|0.641|0.8562|0.9143|97837|
0.3821|0.6483|0.8679|0.9275|146463|
0.3848|0.6532|0.8744|0.9348|195309|
"""
data_noterms_marco = []
for i,line in enumerate(noterms_marco.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[2])
    posting_length = int(fields[-2])
    data_noterms_marco.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "w.o. Terms",
        "portion": posting_length / all_doc_count["marco"],
        "inverse-portion": round(all_doc_count["marco"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["marco"]*48),
    })
data_noterms_marco = pd.DataFrame(data_noterms_marco)


notopics_marco = """
0.3539|0.5835|0.7409|0.7639|9118|
0.3854|0.6485|0.8417|0.8771|22698|
0.3934|0.6706|0.8809|0.9256|48489|
0.3974|0.678|0.8969|0.9481|84619|
0.3977|0.681|0.9046|0.9594|130108|
0.3984|0.6835|0.9125|0.9715|174034|
0.3986|0.6843|0.9134|0.9744|232734|
"""
data_notopics_marco = []
for i,line in enumerate(notopics_marco.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[2])
    posting_length = int(fields[-2])
    data_notopics_marco.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "w.o. Topics",
        "portion": posting_length / all_doc_count["marco"],
        "inverse-portion": round(all_doc_count["marco"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["marco"]*48),
    })
data_notopics_marco = pd.DataFrame(data_notopics_marco)


nodistill_marco = """
0.3658|0.6346|0.8357|0.893|0.05|9178|
0.3758|0.6532|0.8792|0.938|0.15|26798|
0.3811|0.6651|0.8993|0.963|0.23|37698|
0.382|0.6688|0.9055|0.9705|0.37|52314|
0.3829|0.6704|0.91|0.9767|0.61|83345|
0.383|0.671|0.9117|0.9787|0.76|101568|
0.383|0.671|0.9141|0.9816|1.05|217734|
"""
data_nodistill_marco = []
for i,line in enumerate(nodistill_marco.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[2])
    posting_length = int(fields[5])
    data_nodistill_marco.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "w.o. Distill",
        "portion": posting_length / all_doc_count["marco"],
        "inverse-portion": round(all_doc_count["marco"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["marco"]*48),
    })
data_nodistill_marco = pd.DataFrame(data_nodistill_marco)


notrain_marco = """
0.3316|0.5662|0.7452|0.7873|0.05|9549|
0.3547|0.6123|0.8293|0.8741|0.19|35233|
0.3736|0.6484|0.874|0.9344|0.28|50924|
0.3775|0.6608|0.8908|0.954|0.38|69764|
0.381|0.6664|0.9017|0.9667|0.54|99150|
0.3822|0.6689|0.9065|0.9731|0.71|131655|
0.383|0.6716|0.9112|0.9786|1.15|212753|
"""
data_notrain_marco = []
for i,line in enumerate(notrain_marco.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[2])
    posting_length = int(fields[5])
    data_notrain_marco.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "w.o. Train",
        "portion": posting_length / all_doc_count["marco"],
        "inverse-portion": round(all_doc_count["marco"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["marco"]*48),
    })
data_notrain_marco = pd.DataFrame(data_notrain_marco)


ivfopq_nq = """
0.544|0.5953|0.6388|0.715|30937|
0.623|0.674|0.708|0.7789|93482|
0.6548|0.7058|0.7396|0.8047|155988|
0.6864|0.7424|0.7748|0.8363|311502|
0.7161|0.769|0.8025|0.8568|620229|
0.741|0.7939|0.8277|0.8792|1529677|
0.7587|0.8125|0.8476|0.8939|2983870|
"""
data_ivfopq_nq = []
for i,line in enumerate(ivfopq_nq.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[3])
    posting_length = int(fields[-2])
    data_ivfopq_nq.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "IVFOPQ",
        "portion": posting_length / all_doc_count["nq"],
        "inverse-portion": round(all_doc_count["nq"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["nq"] *48),
    })
data_ivfopq_nq = pd.DataFrame(data_ivfopq_nq)


distillvq_nq = """
0.6676|0.7222|0.759|0.8202|22351|
0.682|0.7357|0.7726|0.8327|36985|
0.6964|0.751|0.7886|0.8488|87932|
0.7053|0.7615|0.7992|0.8596|160504|
0.7296|0.7814|0.8139|0.869|366498|
0.7413|0.795|0.8288|0.8834|962054|
0.7482|0.8019|0.8371|0.892|2411963|
"""
data_distillvq_nq = []
for i,line in enumerate(distillvq_nq.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[3])
    posting_length = int(fields[-2])
    data_distillvq_nq.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "Distill-VQ",
        "portion": posting_length / all_doc_count["nq"],
        "inverse-portion": round(all_doc_count["nq"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["nq"] *48),
    })
data_distillvq_nq = pd.DataFrame(data_distillvq_nq)

bivfpq_nq = """
0.7421|0.7895|0.8172|0.8607|0.05|22951|
0.7698|0.8205|0.8488|0.8909|0.13|55693|
0.7778|0.8285|0.8582|0.9003|0.21|90197|
0.7795|0.8319|0.8607|0.9042|0.3|132415|
0.7798|0.833|0.8626|0.9064|0.41|178768|
0.7809|0.8335|0.8637|0.9078|0.53|230080|
0.7806|0.8341|0.8643|0.9086|0.89|389082|
"""
data_bivfpq_nq = []
for i,line in enumerate(bivfpq_nq.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[3])
    posting_length = int(fields[-2])
    data_bivfpq_nq.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "Bi-IVFPQ",
        "portion": posting_length / all_doc_count["nq"],
        "inverse-portion": round(all_doc_count["nq"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["nq"] *48),
    })
data_bivfpq_nq = pd.DataFrame(data_bivfpq_nq)


noterms_nq = """
0.7291|0.7745|0.8047|0.8501|26782|
0.7416|0.7875|0.8177|0.8609|44935|
0.7546|0.8039|0.8335|0.8762|89768|
0.7587|0.8108|0.8388|0.8803|135634|
0.7645|0.8161|0.8432|0.8853|182055|
0.7698|0.8211|0.8493|0.8897|277290|
0.7723|0.8258|0.8532|0.8936|375286|
"""
data_noterms_nq = []
for i,line in enumerate(noterms_nq.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[3])
    posting_length = int(fields[-2])
    data_noterms_nq.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "w.o. Terms",
        "portion": posting_length / all_doc_count["nq"],
        "inverse-portion": round(all_doc_count["nq"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["nq"] *48),
    })
data_noterms_nq = pd.DataFrame(data_noterms_nq)

notopics_nq = """
0.682|0.7258|0.7565|0.805|21847|
0.7488|0.7947|0.8235|0.8648|54351|
0.7657|0.8144|0.8429|0.8867|96569|
0.7715|0.8213|0.851|0.8947|142922|
0.7765|0.8271|0.8582|0.8986|194234|
0.7809|0.833|0.862|0.9055|353236|
0.7825|0.833|0.8623|0.9064|455751|
"""
data_notopics_nq = []
for i,line in enumerate(notopics_nq.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[3])
    posting_length = int(fields[-2])
    data_notopics_nq.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "w.o. Topics",
        "portion": posting_length / all_doc_count["nq"],
        "inverse-portion": round(all_doc_count["nq"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["nq"] *48),
    })
data_notopics_nq = pd.DataFrame(data_notopics_nq)


nodistill_nq = """
0.7338|0.7837|0.8169|0.8567|0.05|23051|
0.7598|0.8139|0.8488|0.8856|0.13|55693|
0.7698|0.823|0.8565|0.8948|0.21|90197|
0.7701|0.8241|0.8582|0.8985|0.3|132415|
0.7706|0.826|0.8607|0.9009|0.41|178768|
0.772|0.8274|0.8615|0.9027|0.53|230080|
0.7723|0.8271|0.8626|0.9055|0.89|389082|
"""
data_nodistill_nq = []
for i,line in enumerate(nodistill_nq.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[3])
    posting_length = int(fields[-2])
    data_nodistill_nq.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "w.o. Distill",
        "portion": posting_length / all_doc_count["nq"],
        "inverse-portion": round(all_doc_count["nq"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["nq"] *48),
    })
data_nodistill_nq = pd.DataFrame(data_nodistill_nq)

notrain_nq = """
0.6886|0.7402|0.7748|0.823|0.05|23633|
0.7485|0.7997|0.833|0.8806|0.23|101344|
0.7557|0.808|0.8418|0.8875|0.32|141657|
0.7598|0.8147|0.8485|0.8928|0.46|200877|
0.7634|0.818|0.8512|0.8956|0.58|251809|
0.767|0.8222|0.8551|0.8981|0.68|296296|
0.7695|0.8252|0.8593|0.9006|0.87|381999|
"""
data_notrain_nq = []
for i,line in enumerate(notrain_nq.strip().split("\n")):
    fields = line.split("|")
    recall = float(fields[3])
    posting_length = int(fields[-2])
    data_notrain_nq.append({
        "recall": recall,
        "posting_length": posting_length,
        "method": "w.o. Train",
        "portion": posting_length / all_doc_count["nq"],
        "inverse-portion": round(all_doc_count["nq"] / posting_length),
        "acceleration": 768/(posting_length / all_doc_count["nq"] *48),
    })
data_notrain_nq = pd.DataFrame(data_notrain_nq)
