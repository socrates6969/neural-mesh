p='src/store.rs'
s=open(p,encoding='utf-8').read()
old='''        let r = a.sync_docs_folder_with(&folder, "mA", "kb", &far).unwrap();
        assert_eq!(r.tombstones_expired, 0);'''
new='''        a.sync_docs_folder_with(&folder, "mA", "kb", &far).unwrap();
        assert!(fs::read_to_string(a.kdir("kb").join("tombstones.jsonl")).unwrap().contains("\\"y\\""), "local legacy tombstone kept");
        assert_eq!(a.read_dropped("kb").len(), 1);
        assert_eq!(ids(&a), vec!["x", "z"], "and y is not re-imported");'''
assert old in s
open(p,'w',encoding='utf-8').write(s.replace(old,new))
