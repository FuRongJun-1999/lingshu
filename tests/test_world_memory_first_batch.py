"""First batch: shared identity, real visual observations, and memory round trips."""
import os
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.brain_store import connect, ingest_scene_to_brain, load_world_from_brain
from lingshu.world.scene_model import WorldModel, ingest_scene, load_world_from_memory
from lingshu.world.wm_simloop import SimulationLoop
from lingshu.world.vprim import VPrim
from lingshu.world.world3d import Camera3D, World3D
from lingshu.world.world_model import UnifiedWorldModel, WMNode


@pytest.fixture
def brain_factory(tmp_path):
    path = os.environ.get('MDCG_BRAIN_PYTHONPATH')
    if not path:
        pytest.skip('Real MCP round trip needs MDCG_BRAIN_PYTHONPATH (CI supplies the pinned brain)')
    agents = []

    def reopen():
        agent = connect(root=str(tmp_path / 'brain'), pythonpath=path,
                        extra_env={'MDCG_LEGACY_ENV_AUTH': '1', 'MDCG_CAN_ADMIN': '1',
                                   'MDCG_LEGACY_ENV_ADMIN': '1', 'MDCG_SUSTAIN': '0',
                                   'MDCG_AUX_ROOT': str(tmp_path / 'brain_aux')})
        agents.append(agent)
        return agent

    yield reopen
    for agent in agents:
        agent.store.client.close()


def scene_with_relations():
    scene = WorldModel()
    scene.add_entity('table', 'table', (3, .45, 5), eid='table-1')
    scene.add_entity('cup', 'cup', (4, 1, 5), eid='cup-1',
                     conditions={'source': 'camera-left', 'timestamp': 100})
    scene.add_entity('chair', 'chair', (8, .45, 5), eid='chair-1')
    scene.relation('cup', 'right_of', 'table').relation('chair', 'near', 'table')
    return scene


def check_restored_scene(restored):
    assert restored.entities['cup'].eid == 'cup-1'
    assert restored.entities['cup'].conditions == {'source': 'camera-left', 'timestamp': 100}
    assert restored.entities['cup'].relations == [('table', 'right_of')]
    assert restored.entities['chair'].relations == [('table', 'near')]
    restored.entities['table'].pos = (6, .45, 5)
    restored.build()
    assert restored.entities['cup'].pos[0] > 6
    assert restored.entities['chair'].pos == pytest.approx((6.8, .45, 5.3))


def test_scene_relations_round_trip_existing_store(tmp_path):
    engine = SpacetimeMemoryEngine(db_path=str(tmp_path / 'scene.db'))
    try:
        agent = SimpleNamespace(engine=engine, store=engine.store)
        ids = ingest_scene(agent, scene_with_relations())
        assert len(ids) == 3
        check_restored_scene(load_world_from_memory(engine))
        assert len(ingest_scene(agent, 'cup|cup|4,1,5|neutral\ncup|cup|4,1,5|neutral')) == 2
    finally:
        engine.store.conn.close()


def test_scene_relations_round_trip_real_brain(brain_factory):
    first = brain_factory()
    assert len(ingest_scene_to_brain(first, scene_with_relations())) == 3
    first.store.client.close()
    check_restored_scene(load_world_from_brain(brain_factory()))


def test_experience_real_brain_restart_preserves_predictive_relations(brain_factory):
    wm = UnifiedWorldModel(size=80)
    visual = World3D()

    def capture(tick):
        image = Image.new('RGB', (640, 480))
        draw = ImageDraw.Draw(image)
        x = 380 + 12 * tick
        draw.rectangle((x, 214, x + 63, 285), fill=(255, 0, 0))
        draw.rectangle((530, 214, 593, 285), fill=(0, 255, 0))
        for eid, channel in [('a', 'R'), ('b', 'G')]:
            vp = VPrim('chair', image.getchannel(channel).getbbox(), eid=eid,
                       source='controlled-pixels', ts=100 + tick)
            visual.add_vprim(vp, 640, 480, tick=tick)
        return visual.observations(tick=tick)

    for tick in range(1, 6):
        wm.perceive(capture(tick), tick=tick, timestamp=100 + tick, tool='camera')
        if tick == 4:
            wm.generate()
    assert (wm.verify()['hits'], wm.verify()['total']) == (2, 2)
    expected = wm.generate()
    assert any(e.source == 'a' and e.target == 'b' and e.relation == 'seek' for e in wm.edges)
    assert expected['predictions']['a']['mode'] == 'exact'
    assert expected['predictions']['a']['predicted'][0] > wm.nodes['a'].pos[0]
    # Larger than the brain's 50 KiB preview: restoration must consume continuation pages.
    wm._conditions['a']['calibration_notes'] = 'x' * 52000
    wm.nodes['unconfirmed'] = WMNode('unconfirmed', 'hidden_target', (20, 1.5, 20),
                                    attrs={'hypothesis': True})
    loop = SimulationLoop(world_model=wm)
    first = brain_factory()
    scene = WorldModel()
    for eid in ['a', 'b']:
        scene.add_entity(eid, 'chair', wm.nodes[eid].pos, eid=eid,
                         conditions={'source': 'controlled-pixels'})
    scene.relation('a', 'left_of', 'b')
    assert len(ingest_scene_to_brain(first, scene)) == 2
    saved = loop.flush_payloads(brain=first)
    assert saved['memory_id']
    first.store.client.close()

    # A prior carries its model/world size; restore into a matching physical scene.
    resumed = SimulationLoop(size=80)
    second = brain_factory()
    restored_scene = load_world_from_brain(second)
    assert {e.eid for e in restored_scene.entities.values()} == {'a', 'b'}
    assert restored_scene.entities['a'].relations == [('b', 'left_of')]
    loaded = resumed.load_priors(brain=second)
    assert loaded['memory_id'] == saved['memory_id']
    assert loaded['seeded'] == 2
    assert set(resumed.wm.nodes) == {'a', 'b'}
    assert not resumed.scene.entities
    assert resumed.wm._conditions['a']['timestamp'] == 105
    assert len(resumed.wm._conditions['a']['calibration_notes']) == 52000
    exact = SimulationLoop(size=80)
    assert exact.load_priors(brain=brain_factory(), memory_id=saved['memory_id'])['seeded'] == 2
    assert resumed.wm.history == wm.history
    assert [e.to_dict() for e in resumed.wm.edges] == [e.to_dict() for e in wm.edges]
    assert resumed.wm.generate() == expected
    assert resumed.wm.verify()['total'] == 0
    resumed.wm.perceive(capture(6), tick=6, tool='camera')
    assert (resumed.wm.verify()['hits'], resumed.wm.verify()['total']) == (2, 2)


def test_explicit_multiview_identity_survives_crossing_and_occlusion():
    visual = World3D()
    wm = UnifiedWorldModel()
    positions = [{'left': (-.4, .5, 5), 'right': (.4, .5, 5)},
                 {'left': (.4, .5, 5), 'right': (-.4, .5, 5)},
                 {'left': (.8, .5, 5)},
                 {'left': (1.2, .5, 5), 'right': (-.8, .5, 5)}]
    for tick, points in enumerate(positions, 1):
        for eid, point in points.items():
            for eye in ((-2, 1.2, 0), (2, 1.2, 0)):
                camera = Camera3D.look_at(eye, (0, .5, 5))
                x, y = camera.project(point, 640, 480)
                visual.add_view('chair', (x-5, y-5, x+5, y+5), 640, 480,
                                camera=camera, eid='' if tick == 1 and eye[0] == 2 else eid, tick=tick)
        observed = visual.observations(tick=tick)
        assert {o['eid'] for o in observed} == set(points)
        assert len(visual.objects) == 2
        wm.perceive(observed, tick=tick, tool='multiview')
        for eid, point in points.items():
            assert wm.nodes[eid].pos == pytest.approx(point, abs=.01)
        if tick == 3:
            assert wm.nodes['right'].last_seen == 2
            assert 'right' not in wm.history[-1]['entities']
    assert {o['eid'] for o in visual._mv.fused_objects()} == {'left', 'right'}


def test_core_visual_pixels_predict_then_verify_at_real_sample_tick(tmp_path):
    engine = SpacetimeMemoryEngine(db_path=str(tmp_path / 'vision.db'))
    try:
        engine.world_model('init')
        engine.wm_simloop('attach')
        assert engine._simloop.wm is engine._wmodel
        decoy = engine.world_model('entity', {'category': 'decoy', 'pos': [20, 1.5, 20]})['entity_id']
        for tick in (0, 3, 6, 9, 12):
            image = Image.new('RGB', (640, 480))
            x = 380 + 2 * tick
            ImageDraw.Draw(image).rectangle((x, 214, x + 63, 285), fill='white')
            engine.world3d('add', {'category': 'chair', 'bbox': image.getbbox(),
                                 'screen_w': 640, 'screen_h': 480, 'eid': 'chair-1',
                                 'source': 'controlled-pixels', 'tick': tick, 'timestamp': 100 + tick})
            result = engine.world_model('perceive_visual', {'tick': tick, 'timestamp': 100 + tick})
            assert result['perceive']['observed'] == 1
            if tick == 9:
                engine.world_model('generate', {'horizon': 3})
        verified = engine.world_model('verify')['verify']
        assert (verified['hits'], verified['total']) == (1, 1)
        assert set(engine._wmodel.nodes) == {'chair-1'}
        assert decoy not in engine._wmodel._obs_snapshot
        assert engine._wmodel._conditions['chair-1']['source'] == 'controlled-pixels'
        assert [frame['tick'] for frame in engine._wmodel.history] == [0, 3, 6, 9, 12]
        assert len(engine._world3d.objects) == 1
    finally:
        engine.store.conn.close()


def test_core_forwards_external_observations_and_reuses_anonymous_identity(tmp_path):
    engine = SpacetimeMemoryEngine(db_path=str(tmp_path / 'observations.db'))
    try:
        for tick, x in enumerate((2, 3, 4), 1):
            result = engine.world_model('perceive', {'observations': [
                {'eid': None, 'category': 'cart', 'pos': (x, 1.5, 2), 'conditions': {'source': 'external'}}],
                'tick': tick, 'tool': 'sensor'})
            assert result['perceive']['observed'] == 1
        eid = result['perceive']['observations'][0]['eid']
        assert eid != 'None'
        engine.world_model('generate')
        result = engine.world_model('perceive', {'observations': [{'category': 'cart', 'pos': (5, 1.5, 2)}],
                                               'tick': 4})
        assert result['perceive']['observations'][0]['eid'] == eid
        assert result['perceive']['consistent'] == 1
        assert len(engine._wmodel.nodes) == 1
    finally:
        engine.store.conn.close()
