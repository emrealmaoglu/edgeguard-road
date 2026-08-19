# EdgeGuard-Road — Kurulum Talimatları

Bu belge, teslim paketindeki kaynak kodun çalışır hâle getirilmesini anlatır. Üç ayrı
kurulum senaryosu vardır ve **hepsini yapmanız gerekmez**; hangisini istediğinize göre
seçin.

| Senaryo | Ne yapmak istiyorsanız | Gereken |
|---|---|---|
| A | Sunum panelini çalıştırmak, sonuçları incelemek | Yalnızca Python |
| B | Testleri çalıştırmak, kodu incelemek | Python + geliştirme paketleri |
| C | Ölçümleri yeniden üretmek, model eğitmek | Python + PyTorch + veri setleri + GPU |

Çoğu inceleme için **Senaryo A yeterlidir** ve birkaç dakika sürer.

---

## Ortak ön koşullar

- **Python 3.10 veya daha yenisi.** Sürümü doğrulayın:

```
python3 --version
```

- **Disk alanı:** Senaryo A için yaklaşık 200 MB, Senaryo C için veri setleri hariç
  birkaç GB.
- **İşletim sistemi:** Linux, macOS veya Windows. Proje Linux (Colab), macOS ve Jetson
  üzerinde (aarch64 / JetPack 6) çalıştırılmıştır.

Kaynak kodu teslim paketindeki `Kaynak_Kod/edgeguard-road/` klasöründedir. Aşağıdaki
komutlar bu klasörün içinden çalıştırılır.

---

## Senaryo A — Sunum panelini çalıştırma

Panel, projenin bütün ölçüm sonuçlarını gösteren Streamlit uygulamasıdır. **Model
ağırlığı, veri seti veya GPU gerektirmez**; gösterdiği her sayıyı diskteki hazır ölçüm
kayıtlarından okur.

### 1. Sanal ortam oluşturun

```
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell kullanıyorsanız son satır yerine:

```
.venv\Scripts\Activate.ps1
```

### 2. Gerekli paketleri kurun

```
python -m pip install --upgrade pip
python -m pip install -e ".[rescue]"
```

Bu adım NumPy, Pillow, PyYAML, Pydantic, Streamlit ve Matplotlib kurar. PyTorch
**kurulmaz** ve panel için gerekli değildir.

### 3. Paneli başlatın

Ölçüm kayıtları teslim paketinde `Uygulama/Panel_Sonuc_Paketi/` klasöründedir. Bu
klasörün yolunu `EDGEGUARD_RESULTS` değişkeniyle verin:

```
EDGEGUARD_RESULTS=/teslim/yolu/Uygulama/Panel_Sonuc_Paketi streamlit run presentation_app.py
```

Windows PowerShell'de:

```
$env:EDGEGUARD_RESULTS="C:\teslim\yolu\Uygulama\Panel_Sonuc_Paketi"
streamlit run presentation_app.py
```

Tarayıcı otomatik açılmazsa `http://localhost:8501` adresine gidin.

Panelin kullanımı `Kullanim_Kilavuzu.pdf` belgesinde anlatılmaktadır.

---

## Senaryo B — Testleri çalıştırma

Senaryo A'daki sanal ortama ek olarak geliştirme paketlerini kurun:

```
python -m pip install -e ".[dev,rescue]"
```

Test paketini çalıştırın:

```
python -m pytest tests/unit -q
```

Depoda 790 birim testi bulunmaktadır. Testlerin bir bölümü proje boyunca gerçekten
karşılaşılmış hatalara karşılık gelir ve o hatanın koşulunu sabitler; bunlar tezin
3. bölümünde anlatılmaktadır.

Kod kalitesi denetimleri:

```
python -m ruff check .
python -m mypy src
```

---

## Senaryo C — Ölçümlerin yeniden üretilmesi

Bu senaryo veri setlerinin ayrı olarak edinilmesini gerektirir. **Veri setleri teslim
paketinde bulunmaz**; lisansları yeniden dağıtıma izin vermez. Ayrıntı için
`Lisans_Bilgileri.txt` dosyasına bakınız.

### Gereken veri setleri

| Veri seti | Kullanım | Edinme |
|---|---|---|
| Cityscapes (fine) | Doğruluk, kalibrasyon | cityscapes-dataset.com — hesap ve şartların kabulü gerekir |
| ACDC | Olumsuz koşul dayanıklılığı | acdc.vision.ee.ethz.ch |
| RoadAnomaly | Open-set değerlendirmesi | İlgili yayının resmî dağıtımı |
| IDD20K | Çok-domainli eğitim | idd.insaan.iiit.ac.in |

### Derin öğrenme yığını

PyTorch, MMEngine, MMCV ve MMSegmentation **çekirdek bağımlılık değildir** ve bilerek
`pyproject.toml` dışında tutulmuştur; CUDA tekerlekleri platforma özgüdür ve platformdan
bağımsız sabitlenemez. Kurulum çalışma zamanında yapılır:

```
python scripts/train/install_semantic_stack.py
```

Sürüm sabitlemeleri `requirements/colab-py311-cu121.lock` dosyasındadır.

### Veri denetimi

Herhangi bir eğitimden önce veri denetimi çalıştırılmalıdır:

```
python scripts/audit_dataset.py \
  --dataset cityscapes \
  --dataset-root /veri/yolu/cityscapes \
  --output-root /cikti/yolu/audit
```

Denetim geçmeden ve bölünme insan tarafından incelenip `--freeze-approved` ile
dondurulmadan eğitim başlatılmaz. Bu kısıt kod düzeyinde uygulanır.

### Uç cihaz ölçümleri

Uç cihaz ölçümleri NVIDIA Jetson Orin Nano Super üzerinde yapılmalıdır ve host makinede
yeniden üretilemez. Sıra: güç modunun 25 W'a sabitlenmesi, ONNX grafiğinden TensorRT
motorunun kurulması, telemetri kaydının başlatılması, sürdürülen yük ölçümü ve
telemetrinin durdurulması. Ayrıntı tezin EK 3 bölümündedir.

Çalışma zamanı sürümleri: TensorRT 10.3.0, JetPack 6 (L4T 36, aarch64), güç modu 25 W,
rastgelelik tohumu 20260728, dağıtım çözünürlüğü 512×1024.

---

## Sorun giderme

**`ModuleNotFoundError: No module named 'edgeguard'`**
Paket kurulmamıştır. Sanal ortamın etkin olduğundan emin olun ve `pip install -e .`
komutunu depo kökünden çalıştırın.

**`ModuleNotFoundError: No module named 'scripts'`**
Betiği depo kökünden çalıştırdığınızdan emin olun. Bu hata proje boyunca gerçekten
karşılaşılmış bir hatadır ve düzeltilmiştir; `scripts/` altındaki 69 betiğin tamamı
otomatik olarak içe aktarma bütünlüğü açısından sınanmaktadır.

**Panel açılıyor ama sayfalar boş**
`EDGEGUARD_RESULTS` yanlış bir klasörü gösteriyordur. Panelin kenar çubuğu okuduğu
klasörün yolunu yazar; oradan doğrulayın. Klasörün içinde `accuracy/`, `acdc/`,
`open_set/`, `jetson/` gibi alt klasörler bulunmalıdır.

**Streamlit "port already in use" diyor**
Başka bir panel örneği çalışıyordur. Farklı bir port verin:
`streamlit run presentation_app.py --server.port 8899`
