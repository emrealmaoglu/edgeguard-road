# EdgeGuard-Road — BM498 Mezuniyet Tezi teslim paketi

**Kaynak Kısıtlı Uç Cihazlarda Belirsizlik Farkındalıklı Açık Küme Yol Tehlikesi
Algılama ve Bağlamsal Risk Analizi**

Düzce Üniversitesi · Mühendislik Fakültesi · Bilgisayar Mühendisliği Bölümü

---

## Dizin yapısı

```
/
├── BM498_Mezuniyet_Tezi_Yunus_Emre_Almaoglu.pdf   ← tezin tam metni
├── OKU_BENI.md                                     ← bu dosya
├── Sunumlar/
│   ├── 00_SUNUMLAR_HAKKINDA.pdf                    ← sunum sayısı açıklaması
│   └── 01_Donem_Baslangic_Sunumu.pdf
├── Kaynak_Kod/
│   └── edgeguard-road/                             ← geliştirilen tüm kaynak kod
├── Uygulama/
│   ├── Demo_Videosu/
│   │   └── EdgeGuard_Sistem_Calisma_Videosu.mp4
│   └── Panel_Sonuc_Paketi/                         ← panelin okuduğu ölçüm kayıtları
└── Dokumanlar/
    ├── Kullanim_Kilavuzu.pdf
    ├── Kurulum_Talimatlari.pdf
    └── Lisans_Bilgileri.txt
```

## Nereden başlamalı

Projeyi **çalıştırmadan** incelemek isteyen biri için en kısa yol:

1. `BM498_Mezuniyet_Tezi_Yunus_Emre_Almaoglu.pdf` — çalışmanın tamamı.
2. `Uygulama/Demo_Videosu/` — sistemin hareketli sahnede çıktısı, kurulum gerektirmez.
3. `Sunumlar/01_Donem_Baslangic_Sunumu.pdf` — projenin başlangıçtaki tasarımı.

Projeyi **çalıştırarak** incelemek için `Dokumanlar/Kurulum_Talimatlari.pdf` içindeki
**Senaryo A** izlenmelidir; birkaç dakika sürer, GPU ya da veri seti gerektirmez ve
projenin bütün ölçüm sonuçlarını gezilebilir hâlde açar.

## Bu pakette bilerek bulunmayanlar

| Bulunmayan | Neden |
|---|---|
| Veri setleri (Cityscapes, ACDC, RoadAnomaly, IDD20K) | Lisansları yeniden dağıtıma izin vermiyor |
| Eğitilmiş model ağırlıkları | Kaynak deponun lisansının ağırlıkları kapsadığı iddia edilemiyor |
| İkinci sunumun slayt dosyası | O sunum Streamlit uygulaması üzerinden yapıldı; slayt üretilmedi |

Ayrıntılar `Dokumanlar/Lisans_Bilgileri.txt` ve `Sunumlar/00_SUNUMLAR_HAKKINDA.pdf`
dosyalarındadır. Ölçüm kayıtları kullanılan ağırlığın sha256 özetini taşır; bu sayede
hangi ağırlığın ölçüldüğü, ağırlık paylaşılmadan doğrulanabilir.

## Çalışmanın doğrulanabilirliği

- Tezde geçen dört ondalıklı her sayının bir ölçüm kaydında karşılığı olduğu otomatik
  olarak denetlenmiştir (`scripts/verify_reported_numbers.py`).
- Depoda 790 birim testi bulunmaktadır; bir bölümü proje boyunca gerçekten karşılaşılmış
  hatalara karşılık gelir.
- Proje boyunca yapılan her materyal değişiklik, doğrulaması ve gerekçesiyle birlikte
  ekle-sadece bir kayıtta tutulmuştur: `Kaynak_Kod/edgeguard-road/docs/AI_USAGE_LOG.md`.
- Ölçülmeyen şeyler "ölçülmedi" diye yazılmıştır ve geri çekilen iki sonuç raporda
  durmaktadır; silinmemiştir.
