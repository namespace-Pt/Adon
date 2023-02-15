import jsonlines


all_positives = []
all_queries = []
with open("/share/project/peitian/Code/Uni-Retriever/src/data/cache/NQ/dataset/query/openai/requests_results.jsonl", "r+", encoding="utf8") as f:
    for i, item in enumerate(jsonlines.Reader(f)):
        tidx = int(item[0]["user"])
        queries = item[1]["choices"][0]["text"].replace("\n", " ").split(" --")[1:]
        all_positives.append(tidx)
        all_queries.append(queries)

query_file = open("/share/project/peitian/Data/NQ/queries.openai.tsv", "w")
qrel_file = open("/share/project/peitian/Data/NQ/qrels.openai.tsv", "w")

query_idx = 0
for i, queries in enumerate(all_queries):
    for j, query in enumerate(queries):
        query_line = "\t".join([str(query_idx), query.lower()]) + "\n"
        qrel_line = "\t".join([str(query_idx), "0", str(all_positives[i]), "1"]) + "\n"
        query_file.write(query_line)
        qrel_file.write(qrel_line)
        query_idx += 1

query_file.close()
qrel_file.close()


all_positives = []
all_queries = []
with open("/share/project/peitian/Code/Uni-Retriever/src/data/cache/NQ/dataset/query/openai/requests.structured_results.jsonl", "r+", encoding="utf8") as f:
    for i, item in enumerate(jsonlines.Reader(f)):
        tidx = int(item[0]["user"])
        queries = item[1]["choices"][0]["text"].replace("\n", " ").split(" --")[1:]
        all_positives.append(tidx)
        all_queries.append(queries)

query_file = open("/share/project/peitian/Data/NQ/queries.openai2.tsv", "w")
qrel_file = open("/share/project/peitian/Data/NQ/qrels.openai2.tsv", "w")

query_idx = 0
for i, queries in enumerate(all_queries):
    for j, query in enumerate(queries):
        query_line = "\t".join([str(query_idx), query.lower()]) + "\n"
        qrel_line = "\t".join([str(query_idx), "0", str(all_positives[i]), "1"]) + "\n"
        query_file.write(query_line)
        qrel_file.write(qrel_line)
        query_idx += 1

query_file.close()
qrel_file.close()
