export ETCD_VER=v3.6.7
export ETCD_IMAGE=gcr.io/etcd-development/etcd:${ETCD_VER}

docker run -d --name etcd \
  -p 2379:2379 -p 2380:2380 \
  ${ETCD_IMAGE} \
  /usr/local/bin/etcd \
  --name node1 \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://127.0.0.1:2380 \
  --initial-cluster node1=http://127.0.0.1:2380 \
  --initial-cluster-state new \
  --data-dir /etcd-data
