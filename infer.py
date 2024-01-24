import argparse

from cdl import data
from cdl.mf import MatrixFactorizationModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collaborative Deep Learning inference')
    parser.add_argument('--topk', type=int, default=300)
    parser.add_argument('--model', default='model.pt')
    args = parser.parse_args()

    mfm = MatrixFactorizationModel((1, 1), 1)
    data.load_model(sdae=None, mfm=mfm, filename=args.model)

    ratings_test_dataset = data.load_cf_test_data()

    recall = mfm.compute_recall(ratings_test_dataset.to_dense(), args.topk)
    print(f'recall@{args.topk}: {recall.item()}')