import time
import etcd3
from typing import List, Optional, Tuple

PREFIX = "/demo_kvcache"


def owner_key(tag: str, owner_id: str) -> str:
    return f"{PREFIX}/tag/{tag}/owners/{owner_id}"


def owners_prefix(tag: str) -> str:
    return f"{PREFIX}/tag/{tag}/owners/"


def publish(etcd: etcd3.Etcd3Client, tags: List[str], owner_id: str, lease: etcd3.Lease) -> None:
    """set/publish: 把 owner 加入每个 tag 的 owners 集合，并绑定到 owner 的 lease 上"""
    for t in tags:
        etcd.put(owner_key(t, owner_id), "1", lease=lease)


def unpublish(etcd: etcd3.Etcd3Client, tags: List[str], owner_id: str) -> None:
    """del: 从每个 tag 的 owners 集合里移除该 owner"""
    for t in tags:
        etcd.delete(owner_key(t, owner_id))


def pick_any_owner(etcd: etcd3.Etcd3Client, tag: str) -> Optional[str]:
    """从某个 tag 的 owners 集合里随便挑一个 owner（如果没有返回 None）"""
    # get_prefix 会返回 (value, meta) 的迭代器；meta.key 是 bytes
    for _val, meta in etcd.get_prefix(owners_prefix(tag)):
        k = meta.key.decode("utf-8")
        # k: /.../tag/<tag>/owners/<owner_id>
        owner_id = k.split("/")[-1]
        return owner_id
    return None


def get_last_hit(etcd: etcd3.client, tags: List[str]) -> Tuple[Optional[str], int]:
    """
    get: 从后往前找最后一个能查到 owners 的 tag。
    返回 (owner_id, num_tags). num_tags = 命中前缀长度
    """
    for i in range(len(tags) - 1, -1, -1):
        t = tags[i]
        owner = pick_any_owner(etcd, t)
        if owner is not None:
            return owner, i + 1
    return None, 0


def list_owners(etcd: etcd3.Etcd3Client, tag: str) -> List[str]:
    """辅助打印：列出某个 tag 当前所有 owners"""
    res = []
    for _val, meta in etcd.get_prefix(owners_prefix(tag)):
        k = meta.key.decode("utf-8")
        res.append(k.split("/")[-1])
    return sorted(res)


def cleanup_prefix(etcd: etcd3.Etcd3Client) -> None:
    """清理 demo 使用的所有 key（可选）"""
    # etcd3-py 没有直接 delete_prefix 的统一接口，这里用 get_prefix 找到再删
    for _val, meta in etcd.get_prefix(PREFIX + "/"):
        etcd.delete(meta.key)


def main():
    etcd = etcd3.client(host="127.0.0.1", port=2379)

    # 可选：清理上一次 demo 残留
    cleanup_prefix(etcd)

    # 模拟两台推理引擎（两个 owner）
    ownerA = "10.0.0.1:9000#epochA"
    ownerB = "10.0.0.2:9000#epochB"

    # tags（你实际里是 block_hash，这里用短字符串）
    tagsA = ["a", "b", "c"]      # ownerA 拥有 a,b,c
    tagsB = ["a", "b"]           # ownerB 拥有 a,b
    query = ["a", "b", "c", "d"] # 查询：d 不存在

    # 关键点：每个 owner 申请自己的 lease
    leaseA = etcd.lease(5)   # TTL=5s（故意短，方便演示过期）
    leaseB = etcd.lease(30)  # TTL=30s（保持更久）

    # set/publish
    publish(etcd, tagsA, ownerA, leaseA)
    publish(etcd, tagsB, ownerB, leaseB)

    print("== After publish ==")
    print("owners(a) =", list_owners(etcd, "a"))
    print("owners(b) =", list_owners(etcd, "b"))
    print("owners(c) =", list_owners(etcd, "c"))

    # get：从后往前找最后一个命中的 tag
    owner, n = get_last_hit(etcd, query)
    print(f"get_last_hit({query}) => owner={owner}, num_tags={n}")
    # 期望：命中 c => ownerA, num_tags=3

    # 不对 leaseA 做 keepalive，让它过期
    print("\n== Sleep 6s to let leaseA expire (TTL=5s) ==")
    time.sleep(6)

    print("== After leaseA expired ==")
    print("owners(a) =", list_owners(etcd, "a"))
    print("owners(b) =", list_owners(etcd, "b"))
    print("owners(c) =", list_owners(etcd, "c"), "(should be empty)")

    owner2, n2 = get_last_hit(etcd, query)
    print(f"get_last_hit({query}) => owner={owner2}, num_tags={n2}")
    # 期望：c 的 owners 已空，回退到 b => ownerB, num_tags=2

    # 演示 del/unpublish（主动撤销 ownerB 在 a,b 上的拥有权）
    print("\n== Unpublish ownerB on [a,b] ==")
    unpublish(etcd, ["a", "b"], ownerB)
    print("owners(a) =", list_owners(etcd, "a"))
    print("owners(b) =", list_owners(etcd, "b"))

    owner3, n3 = get_last_hit(etcd, query)
    print(f"get_last_hit({query}) => owner={owner3}, num_tags={n3}")
    # 期望：此时 a,b,c 都没 owners => MISS (None, 0)

    etcd.close()


if __name__ == "__main__":
    main()
