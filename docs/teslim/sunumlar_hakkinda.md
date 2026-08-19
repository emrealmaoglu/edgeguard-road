# Sunumlar hakkında açıklama

Bu klasördeki sunum sayısının teslim şablonunda belirtilen sayıdan az olmasının nedeni
aşağıda açıklanmıştır.

## Dönem yapısı

Bu proje **yaz okulu döneminde** yürütülmüştür. Yaz okulu dönemi normal bir güz/bahar
döneminden kısa olduğu için ders kapsamında **iki sunum** yapılmıştır. Teslim
şablonundaki "en az 3 sunum" beklentisi normal dönem uzunluğuna göre tanımlanmıştır.

## Yapılan sunumlar

### 1. Dönem başlangıç sunumu — `01_Donem_Baslangic_Sunumu.pdf`

Slayt tabanlı sunum; bu klasörde dosya olarak bulunmaktadır. İçeriği: problem tanımı ve
motivasyon, önerilen sistem mimarisi, belirsizlik farkındalığı ve açık küme algılama
yaklaşımı, bağlamsal risk analizi, veri setleri ve domain yapısı, model adayları ve
eleme stratejisi, deney ve değerlendirme protokolü, edge deployment ve Jetson benchmark
planı, o tarihteki proje durumu ve beklenen katkılar.

Bu sunum **öneri ve tasarım aşamasında** yapılmıştır; sunumun kendisi de bunu açıkça
belirtmektedir ("Sonuç iddiası yok — deney aşaması başlıyor", "Henüz final sonuç yok").
Dolayısıyla sunumdaki plan ile tezde raporlanan nihai sonuçlar arasındaki farklar
beklenen farklardır. Sunum o tarihteki durumun kaydıdır ve **geriye dönük
düzeltilmemiştir.**

### 2. Sonuç ve demo sunumu — **slayt dosyası yoktur**

İkinci sunum PowerPoint veya benzeri bir slayt dosyası üzerinden yapılmamıştır. Sunum,
proje kapsamında geliştirilen **Streamlit tabanlı canlı panel uygulaması** üzerinden
gerçekleştirilmiştir. Sunum sırasında panelin sayfaları sırayla gezilerek ölçüm
sonuçları, karşılaştırma çizelgeleri, güvenilirlik diyagramları, open-set bulguları, uç
cihaz telemetrisi ve nitel görseller doğrudan uygulama üzerinden gösterilmiştir.

Bu nedenle bu sunumun bir `.pptx` veya `.pdf` karşılığı **yoktur ve üretilmemiştir.**
Sunum materyalinin kendisi bir uygulamadır ve bu teslimde şu konumlardadır:

- Uygulamanın kaynak kodu: `Kaynak_Kod/edgeguard-road/presentation_app.py`
- Uygulamanın çalıştırılması: `Dokumanlar/Kullanim_Kilavuzu.pdf`
- Sunumda gösterilen sistem çıktısı videosu:
  `Uygulama/Demo_Videosu/EdgeGuard_Sistem_Calisma_Videosu.mp4`

Panel, gösterdiği her sayıyı diskteki ölçüm kayıtlarından okur; elle yazılmış hiçbir
değer içermez. Bu nedenle sunum içeriği, panelin kendisi çalıştırılarak birebir yeniden
üretilebilir.

## Özet

| # | Sunum | Biçim | Bu teslimdeki karşılığı |
|---|---|---|---|
| 1 | Dönem başlangıç sunumu | Slayt (PDF) | `01_Donem_Baslangic_Sunumu.pdf` |
| 2 | Sonuç ve demo sunumu | Canlı Streamlit uygulaması | Uygulama kaynak kodu ve demo videosu |

Gerçekte yapılmamış bir sunumun slaytı sonradan **üretilmemiştir.**
