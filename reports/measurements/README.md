# Ölçüm kayıtları

Tezdeki sayıların arkasındaki **ham kayıtlar**. Buradaki her dosya gerçekten koşmuş bir
ölçümün çıktısıdır; hiçbiri elle yazılmadı, düzenlenmedi veya beklenen değerle
doldurulmadı. Her kayıt kendi `model_sha256`'sını taşır, yani hangi ağırlıklarla
ölçüldüğü dosyanın kendisinden doğrulanabilir.

Bunlar sürüm kontrolünde, çünkü:

1. Tezde geçen bir sayı ile onu üreten kayıt arasındaki bağın kaybolmaması gerekir.
2. Toplam 24 KB — saklamamak için bir sebep yok.
3. Sunum paneli bunları okuyor ve panel Jetson'da çalışıyor; `git pull` iki makine
   arasında dosya taşımaktan basit.

## İçindekiler

| klasör | ne | üreten |
|---|---|---|
| `drivable/` | 5 mimari × 200 Cityscapes val karesi: yol IoU, iki toleransta sınır F1, yanlış-sürülebilir oranı, parçalanma | `scripts/evaluate_drivable_area.py` |
| `temporal/` | 150 ardışık demoVideo karesi: iz ömürleri, zamansal kalıcılığın risk sıralamasına etkisi | `scripts/measure_temporal_persistence.py` |

## Panele yerleştirme (Jetson'da)

```bash
cd ~/edgeguard-road && git pull
cp -r reports/measurements/drivable reports/measurements/temporal ~/eg-presentation/
```

Cihazda üretilen kayıtlar (`jetson/`, `telemetry/`, `profile/`) buraya **kopyalanmaz** —
onlar zaten cihazda, `~/eg-presentation` altında durur ve oradan taşınmalarına gerek yok.

## Burada olmayanlar

- **Jetson gecikme/güç/telemetri kayıtları** — cihazda üretilir ve cihazda kalır
  (`~/results` ve `~/eg-presentation/{jetson,telemetry,profile}`). Büyükler ve tek bir
  makinede anlamlılar.
- **Mühürlü final test verisi üzerinde hiçbir ölçüm.** O kapı yalnızca insan tarafından
  açılır (`docs/adr/0005`) ve açılmadı.
