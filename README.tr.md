# Mevzuat İstihbarat Asistanı

Siber güvenlik yönetişimi ve uyum ekipleri için kaynak destekli mevzuat
soru-cevap sistemi. DORA (ve ileride NIS2, KVKK, BDDK, ISO 27001)
hakkında doğal dilde soru sorun; cevaplar, ilgili maddelere atıfla
birlikte doğrudan mevzuat metnine dayandırılır. **Tamamen yerel**
çalışır — hiçbir veri makineden dışarı çıkmaz.

*([English README](README.md))*

## Mimari

```
EUR-Lex / PDF'ler ──> alım (maddelere ayrıştırma, metadata ile parçalama)
                        │
                        ▼
              ChromaDB (gömme vektörleri) + BM25 anahtar kelime dizini
                        │  hibrit erişim (RRF füzyonu)
                        ▼
              RAG hattı ──> Ollama üzerinden yerel LLM
                        │
                        ▼
          FastAPI arka ucu  +  Streamlit sohbet arayüzü
```

Temel tasarım kararları:

- **Madde-farkında parçalama** — parçalar hiçbir zaman madde sınırını
  aşmaz, böylece her cevap "DORA, Madde 17" gibi doğru bir atıf
  yapabilir.
- **Hibrit erişim** — BM25 tam terimleri yakalar ("Article 5",
  "madde 12"); gömme vektörleri kavramsal soruları yakalar; Reciprocal
  Rank Fusion (RRF) ikisini birleştirir.
- **Karşılaştırma modu** — erişim her mevzuat için ayrı ayrı çalışır
  (metadata ile filtrelenir), böylece iki çerçeve karşılaştırılırken
  ikisi de bağlamda temsil edilir.
- **Çok dilli gömme vektörleri** — Türkçe mevzuatlar (KVKK, BDDK) mimariyi
  değiştirmeden çalışır. Ayrıştırıcı zaten "Madde N" ifadesini tanır.
- **Değiştirilebilir LLM katmanı** — varsayılan olarak Ollama; herhangi
  bir LLM kurulmadan önce erişimi geliştirip test edebilmeniz için
  `EchoLLM` yedek seçeneği bulunur.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

[Ollama](https://ollama.com)'yı kurun ve bir model indirin:

```bash
ollama pull qwen2.5:7b-instruct
```

(Qwen 2.5 hem İngilizce hem Türkçe hukuki metinleri iyi işler. Herhangi
bir Ollama modeli çalışır — `regintel/config.py` içindeki
`OLLAMA_MODEL` değerini değiştirin.)

## Kullanım

```bash
# 1. Mevzuat kaynak metnini indirin (bir kereliğine internet gerekir)
python -m regintel.ingestion.download DORA

# 2. Ayrıştır + indeksle (ilk çalıştırmada gömme modeli indirilir, ~500MB)
python scripts/ingest.py

# 3a. Komut satırından sorun
python scripts/ask.py "ICT olaylarını bildirme süreleri nedir?"
python scripts/ask.py --compare DORA NIS2 "olay bildirim süreleri"

# 3b. Ya da sohbet arayüzünü başlatın
streamlit run ui/app.py

# 3c. Ya da API'yi çalıştırın
uvicorn regintel.api.main:app --reload   # dokümantasyon: http://localhost:8000/docs
```

Sohbet arayüzü tamamen Türkçedir; LLM ise sorunun sorulduğu dilde
cevap verir — İngilizce sorulara İngilizce, Türkçe sorulara Türkçe.

## Mevzuat ekleme

1. `regintel/config.py` içindeki `REGULATIONS` sözlüğüne bir kayıt
   ekleyin.
2. Kaynağı `data/raw/<ID>.html` (EUR-Lex) veya `<ID>.txt` (bir PDF'den
   çıkarılmış düz metin — Türkçe "Madde" başlıkları desteklenir) olarak
   ekleyin.
3. `python scripts/ingest.py <ID>` komutunu çalıştırın.

## Testler

```bash
pytest tests/ -v
```

Testler, ayrıştırma, parçalama, metadata, filtreleme ve karşılaştırma
erişimini her ortamda doğrulamak için sentetik bir örnek mevzuat
kullanarak çevrimdışı çalışır (gömme vektörü veya LLM gerekmez).

## Veri gizliliği

Her bileşen — gömme vektörleri (sentence-transformers), vektör deposu
(ChromaDB), anahtar kelime dizini (BM25), LLM (Ollama) — localhost
üzerinde çalışır. Gereken tek ağ erişimi, mevzuat metinlerinin ve gömme
modelinin bir kereliğine indirilmesidir; bu işlem üretim ortamı
dışında yapılıp içeri kopyalanabilir.

## Ekip için dağıtım (banka sunucusu + SSO)

Ekibinizin herkesin kendi bilgisayarında çalıştırması yerine, kendi
kurumsal kimlik bilgileriyle bağlandığı ortak bir sunucu için:

1. **Ollama sunucuda yerel kalmalı.** `regintel/config.py` içindeki
   `OLLAMA_BASE_URL` zaten `localhost:11434` adresini gösteriyor —
   böyle bırakın. Ollama'nın kendine ait bir kimlik doğrulaması yoktur,
   bu yüzden ağdan asla erişilebilir olmamalı; yalnızca aynı makinedeki
   Streamlit sürecinden erişilmelidir.
2. **Girişi yapılandırın.** Streamlit'in yerleşik kimlik doğrulaması
   (`ui/app.py` içinde kullanılan `st.login`/`st.logout`), bankanızın
   kimlik sağlayıcısına OIDC üzerinden yetki devreder — bu uygulama
   hiçbir şifre saklamaz. `.streamlit/secrets.toml.example` dosyasını
   `.streamlit/secrets.toml` olarak kopyalayıp, BT ekibinizin dahili
   bir uygulama kaydı için vereceği `client_id`/`client_secret`/
   `server_metadata_url` değerlerini doldurun. Bu dosya yoksa uygulama
   **hiçbir giriş kontrolü olmadan** çalışır — kendi bilgisayarınız
   için sorun değil, ortak sunucu için değil.
3. **Önüne TLS koyun.** Streamlit'in kendisi HTTPS sonlandırması
   yapmaz. Bankanın dahili sertifikasıyla bir ters proxy (nginx/Caddy)
   arkasında çalıştırın ve `secrets.toml` içindeki `redirect_uri`
   değerini o HTTPS adresine göre ayarlayın.
4. Her soru/cevap yine yerel olarak `data/chats/*.jsonl` dosyalarına
   kaydedilir; artık kimliği doğrulanmış kullanıcının bilgisiyle
   etiketlenir — bu da kimin ne sorduğuna dair denetim izinizdir.
