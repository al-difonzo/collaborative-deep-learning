train:
	python train.py -v

infer:
	python infer.py

citeulike-a: mkdir-citeulike-a bow-citeulike-a relationships-citeulike-a bert-citeulike-a
citeulike-t: mkdir-citeulike-t bow-citeulike-t relationships-citeulike-t bert-citeulike-t

clean:
	rm -rf data/processed

mkdir-citeulike-a:
	mkdir -p data/processed/citeulike-a

mkdir-citeulike-t:
	mkdir -p data/processed/citeulike-t

bert-citeulike-a:
	python scripts/compute_bert.py citeulike-a

bert-citeulike-t:
	python scripts/compute_bert.py citeulike-t

bow-citeulike-a:
	python scripts/compute_bow.py citeulike-a

bow-citeulike-t:
	python scripts/compute_bow.py citeulike-t

relationships-citeulike-a:
	python scripts/compute_relationships.py citeulike-a

relationships-citeulike-t:
	python scripts/compute_relationships.py citeulike-t
