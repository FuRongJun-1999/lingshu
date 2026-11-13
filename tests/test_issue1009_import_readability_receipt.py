"""Additive diagnostics, not normalization or physical-loss repair (#1009)."""
import json
import pytest
from lingshu.core.core import SpacetimeMemoryEngine

@pytest.fixture
def engines():
    src=SpacetimeMemoryEngine(':memory:'); dst=SpacetimeMemoryEngine(':memory:')
    src.add_perception('规范行',importance=.6)
    yield src,dst
    src.close();dst.close()

def backup(src,path):
    src.export_all(str(path))
    return json.loads(path.read_text())

def ingest(dst,path,data):
    path.write_text(json.dumps(data))
    before=path.read_bytes()
    dst.store.last_skipped_rows=['unchanged-read-state']
    result=dst.import_all(str(path))
    assert path.read_bytes()==before
    assert dst.store.last_skipped_rows==['unchanged-read-state']
    return result

@pytest.mark.parametrize('field',['tags','condition_space'])
def test_explicit_null_reported_without_rewriting(engines,tmp_path,field):
    src,dst=engines; path=tmp_path/'backup.json';data=backup(src,path)
    data['nodes'][0][field]=None; ident=data['nodes'][0]['id']
    result=ingest(dst,path,data)
    assert result['imported']['nodes']==1
    assert dst.store.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]==1
    assert result['unreadable_imported_nodes']=={'count':1,'sample':[ident]}
    assert dst.store.conn.execute('SELECT '+field+' FROM nodes').fetchone()[0] is None
    assert result['integrity_ok'] is True # Existing FK/orphan meaning unchanged.
    assert result['normalized_values']=={} and result['skipped_rows']=={}


def test_valid_roundtrip_ignores_unrelated_existing_corruption(engines,tmp_path):
    src,dst=engines;dst.add_perception('旧坏行',importance=.5)
    dst.store.conn.execute('UPDATE nodes SET tags=NULL');dst.store.conn.commit()
    path=tmp_path/'backup.json';data=backup(src,path)
    result=ingest(dst,path,data)
    assert result['unreadable_imported_nodes']=={'count':0,'sample':[]}
    row=dict(dst.store.conn.execute('SELECT * FROM nodes WHERE id=?',(data['nodes'][0]['id'],)).fetchone())
    assert row==data['nodes'][0]

@pytest.mark.parametrize('last_bad',[False,True])
def test_duplicate_id_reports_final_stored_row_once(engines,tmp_path,last_bad):
    src,dst=engines;path=tmp_path/'backup.json';data=backup(src,path)
    good=data['nodes'][0];bad=dict(good,tags=None)
    data['nodes']=[good,bad] if last_bad else [bad,good]
    result=ingest(dst,path,data)
    assert result['imported']['nodes']==2 # Existing receipt semantics retained.
    assert dst.store.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]==1
    assert result['unreadable_imported_nodes']=={'count':int(last_bad),'sample':[good['id']] if last_bad else []}

def test_null_id_is_not_lost_by_id_lookup(engines,tmp_path):
    src,dst=engines;path=tmp_path/'backup.json';data=backup(src,path)
    data['nodes'][0].update(id=None,tags=None)
    result=ingest(dst,path,data)
    assert result['unreadable_imported_nodes']=={'count':1,'sample':[None]}

def test_sample_bounded_count_complete(engines,tmp_path):
    src,dst=engines;path=tmp_path/'backup.json';data=backup(src,path);row=data['nodes'][0]
    data['nodes']=[dict(row,id=f'bad-{i}',tags=None) for i in range(25)]
    result=ingest(dst,path,data)
    assert result['unreadable_imported_nodes']=={'count':25,'sample':[f'bad-{i}' for i in range(20)]}

def test_empty_import_does_not_scan_old_rows(engines,tmp_path):
    _,dst=engines;dst.add_perception('旧坏行',importance=.5)
    dst.store.conn.execute('UPDATE nodes SET tags=NULL');dst.store.conn.commit()
    result=ingest(dst,tmp_path/'backup.json',{'nodes':[]})
    assert result['unreadable_imported_nodes']=={'count':0,'sample':[]}

@pytest.mark.parametrize('field',['tags','condition_space'])
def test_missing_json_column_preserved_and_reported(engines,tmp_path,field):
    src,dst=engines;path=tmp_path/'backup.json';data=backup(src,path)
    del data['nodes'][0][field]
    result=ingest(dst,path,data)
    assert result['unreadable_imported_nodes']=={'count':1,'sample':[data['nodes'][0]['id']]}
    assert dst.store.conn.execute('SELECT '+field+' FROM nodes').fetchone()[0] is None

def test_mixed_and_rejected_rows_are_not_confused(engines,tmp_path):
    src,dst=engines;path=tmp_path/'backup.json';data=backup(src,path);row=data['nodes'][0]
    data['nodes']=[row,dict(row,id='unreadable',tags=None),dict(row,id='rejected',access_count={})]
    result=ingest(dst,path,data)
    assert result['unreadable_imported_nodes']=={'count':1,'sample':['unreadable']}
    assert result['skipped_rows']=={'nodes':1}
    assert result['imported']['nodes']==2
    assert dst.store.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]==2

def test_diagnostic_sql_failure_rolls_back(engines,tmp_path):
    import sqlite3
    src,dst=engines;path=tmp_path/'backup.json';data=backup(src,path)
    dst.add_perception('既有数据',importance=.5)
    before=list(dst.store.conn.iterdump())
    path.write_text(json.dumps(data))
    def authorizer(action,table,column,db,trigger):
        if action==sqlite3.SQLITE_READ and table=='nodes' and column=='content':
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    dst.store.conn.set_authorizer(authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError): dst.import_all(str(path))
    finally: dst.store.conn.set_authorizer(None)
    assert list(dst.store.conn.iterdump())==before
