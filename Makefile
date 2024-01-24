train:
	python train.py -v

infer:
	python infer.py

clean:
	rm -rf data/processed

citeulike-a: mkdir-citeulike-a bow-citeulike-a relationships-citeulike-a bert-citeulike-a
citeulike-t: mkdir-citeulike-t bow-citeulike-t relationships-citeulike-t bert-citeulike-t
# amz-precomputed-embeddings: mkdir -p data/processed/$DATASET_NAME && transform-amz-file && openai-amz-fashion

mkdir-citeulike-a:
	mkdir -p data/processed/citeulike-a

mkdir-citeulike-t:
	mkdir -p data/processed/citeulike-t

# openai-amz-fashion:
# 	python scripts/compute_openai.py amz-fashion $OPENAI_TOKEN

bert-citeulike-a:
	python scripts/compute_bert.py --dataset citeulike-a

bert-citeulike-t:
	python scripts/compute_bert.py --dataset citeulike-t

# bert-amz-fashion:
# 	python scripts/compute_bert.py --dataset amz-fashion --in_path data/preprocessed/amz-fashion.csv

bow-citeulike-a:
	python scripts/compute_bow.py citeulike-a

bow-citeulike-t:
	python scripts/compute_bow.py citeulike-t

# bow-amz-fashion:
# 	python scripts/compute_bow.py amz-fashion

relationships-citeulike-a:
	python scripts/compute_relationships.py citeulike-a

relationships-citeulike-t:
	python scripts/compute_relationships.py citeulike-t

transform-amz:
	python scripts/transform_amz_file.py
