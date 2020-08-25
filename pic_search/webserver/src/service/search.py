import logging
import os
import os.path as path
from common.config import DEFAULT_TABLE
from common.const import default_cache_dir
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index,has_table
from preprocessor.vggnet import VGGNet
from preprocessor.vggnet import vgg_extract_feat
from diskcache import Cache


def query_name_from_ids(vids):
    res = []
    cache = Cache(default_cache_dir)
    for i in vids:
        if i in cache:
            res.append(cache[i])
    return res


def do_search(table_name, img_path, top_k, model, graph, sess):
    try:
        feats = []
        index_client = milvus_client()
        feat = vgg_extract_feat(img_path, model, graph, sess)
        feats.append(feat)
        _, vectors = search_vectors(index_client, table_name, feats, top_k)
        vids = [x.id for x in vectors[0]]
        # print(vids)
        # res = [x.decode('utf-8') for x in query_name_from_ids(vids)]

        res_id = [x.decode('utf-8') for x in query_name_from_ids(vids)]
        # print(res_id)
        res_distance = [x.distance for x in vectors[0]]
        # print(res_distance)
        # res = dict(zip(res_id,distance))

        return res_id,res_distance
    except Exception as e:
        logging.error(e)
        return "Fail with error {}".format(e)

def do_index(table_name, img_path, model, graph, sess):
    feats = []
    feat = vgg_extract_feat(img_path, model, graph, sess)
    feats.append(feat)
    filename = os.path.split(img_path)[1]
    if not table_name:
        table_name = DEFAULT_TABLE
    cache = Cache(default_cache_dir)
    index_client = milvus_client()
    status, ok = has_table(index_client, table_name)
    if not ok:
        print("create table.")
        create_table(index_client, table_name=table_name)
    print("insert into:", table_name," for image:",filename)
    status, ids = insert_vectors(index_client, table_name, feats)
    cache[ids[0]] = filename.encode()

    print("insert into:", table_name, " for image:", filename," ids:",ids," status:",status)
    create_index(index_client, table_name)
    return ids
