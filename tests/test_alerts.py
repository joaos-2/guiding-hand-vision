from guiding_vision.alerts import AlertEngine, FixedObstacleTracker, is_head_height
from guiding_vision.config import Classes
from guiding_vision.detector import Detection

C = Classes.load()


def det(name, x0, y0, x1, y1, conf=0.8):
    return Detection(C.by_name(name), conf, x0, y0, x1, y1)


def test_placa_suspensa_vira_aviso_na_altura_da_cabeca():
    a = AlertEngine().step([det("placa", 0.4, 0.1, 0.6, 0.35)], now=0)
    assert a is not None
    assert a.text.startswith("placa na altura da cabeça")
    assert "à frente" in a.text


def test_buraco_nunca_e_altura_da_cabeca():
    assert not is_head_height(det("buraco", 0.4, 0.1, 0.6, 0.3))


def test_carro_distante_acima_do_horizonte_nao_e_altura_da_cabeca():
    a = AlertEngine().step([det("carro", 0.0, 0.0, 0.3, 0.23)], now=0)
    assert a is None or "cabeça" not in a.text


def test_pessoa_longe_e_silenciada():
    assert AlertEngine().step([det("pessoa", 0.45, 0.4, 0.55, 0.6)], now=0) is None


def test_pessoa_perto_no_corredor_avisa():
    a = AlertEngine().step([det("pessoa", 0.35, 0.1, 0.65, 0.95)], now=0)
    assert a is not None and "perto" in a.text


def test_rosto_e_placa_de_veiculo_nao_geram_aviso():
    dets = [det("rosto", 0.4, 0.1, 0.6, 0.9), det("placa_veiculo", 0.4, 0.5, 0.6, 0.9)]
    assert AlertEngine().step(dets, now=0) is None


def test_cooldown_evita_repeticao():
    e = AlertEngine(cooldown_s=4)
    d = [det("poste", 0.4, 0.05, 0.55, 0.95)]
    assert e.step(d, now=0) is not None
    assert e.step(d, now=1) is None
    assert e.step(d, now=5) is not None


def test_prioriza_obstaculo_perto_no_corredor():
    dets = [det("lixeira", 0.8, 0.6, 0.9, 0.75), det("poste", 0.4, 0.05, 0.55, 0.95)]
    a = AlertEngine().step(dets, now=0)
    assert a.detection.cls.name == "poste"


def test_tracker_so_reporta_obstaculo_fixo_persistente():
    t = FixedObstacleTracker(window=10, min_hits=5, report_cooldown_s=60)
    d = [det("buraco", 0.4, 0.7, 0.6, 0.9), det("carro", 0.0, 0.3, 0.3, 0.8)]
    out = []
    for i in range(6):
        out += t.update(d, now=i)
    assert [x.cls.api_type for x in out] == ["buraco"]  # carro não é fixo
