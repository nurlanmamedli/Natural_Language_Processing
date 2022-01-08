# BERT model for answering frequently asked questions
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

model_name = 'mys/bert-base-turkish-cased-nli-mean-faq-mnr'

questions = [
    "İzolasyon süresi dolduktan sonra kişi bulaştırıcı mı ?"
]

answers = [
    "Yeni Koronavirüs (SARS-CoV-2) ilk olarak Çin’in Vuhan şehrinde tespit edilen, solunum yolu enfeksiyonu yapan bir virüstür. Yeni Koronavirüs Hastalığı’na (COVID-19) sebep olur. Bu hastalık yüksek ateş, öksürük, nefes darlığı, baş ağrısı, boğaz ağrısı, burun akıntısı, kas ve eklem ağrısı, halsizlik, koku ve tat alma duyusu kaybı, ishal gibi belirtilerle kendini gösterir., Koronavirüsler (CoV) tek zincirli RNA’ya sahip, zarflı, yüzeyinde protein çıkıntıları olan büyük bir virüs ailesidir. Hafif ve orta derecede şiddetli solunum yolu hastalıkların yanı sıra MERS (Orta Doğu Solunum Sendromu), SARS (Şiddetli Akut Solunum Sendromu) ve COVID-19 (Yeni Koronavirüs Hastalığı) gibi şiddetli seyreden hastalıklardan da sorumludur.",
    "COVID-19, hasta kişilerin öksürme, hapşırma veya konuşmayla ortaya saçtığı damlacıkların ortamdaki diğer bireyler tarafından solunması, damlacıkların yapıştığı yüzeylere dokunduktan sonra ellerin ağız, burun veya göze götürülmesiyle bulaşır.",
    "Kesin tanı moleküler testlerle konulmaktadır. Bu testler kişiden alınan boğaz ve/veya burun sürüntüsünde virüsün genetik materyalinin aranması prensibine dayanır",
    "Bu hastalık için geliştirilmiş bir ilaç yoktur. Hastanın genel durumuna göre gerekli destek tedavisi uygulanmaktadır.",
    "Hastalığın etkeni bir virüs türüdür (SARS-CoV-2). Antibiyotikler virüslerin neden olduğu enfeksiyonları önlemek veya tedavi etmek amacıyla kullanılmaz.",
    "Virüsün kuluçka süresi 2 ila 14 gündür.",
    "Hastalık yüksek ateş, öksürük, nefes darlığı, baş ağrısı, boğaz ağrısı, burun akıntısı, kas ve eklem ağrısı, halsizlik, koku ve tat alma duyusu kaybı, ishal gibi belirtilerle kendini gösterir. Belirtisiz vakalar da mevcuttur.",
    "Yeni Koronavirüs (SARS-CoV-2) ilk olarak Çin’in Vuhan şehrinde tespit edilen, solunum yolu enfeksiyonu yapan bir virüstür. Yeni Koronavirüs Hastalığı’na (COVID-19) sebep olur. Bu hastalık yüksek ateş, öksürük, nefes darlığı, baş ağrısı, boğaz ağrısı, burun akıntısı, kas ve eklem ağrısı, halsizlik, koku ve tat alma duyusu kaybı, ishal gibi belirtilerle kendini gösterir., Koronavirüsler (CoV) tek zincirli RNA’ya sahip, zarflı, yüzeyinde protein çıkıntıları olan büyük bir virüs ailesidir. Hafif ve orta derecede şiddetli solunum yolu hastalıkların yanı sıra MERS (Orta Doğu Solunum Sendromu), SARS (Şiddetli Akut Solunum Sendromu) ve COVID-19 (Yeni Koronavirüs Hastalığı) gibi şiddetli seyreden hastalıklardan da sorumludur.",
    "Kesin tanı moleküler testlerle konulmaktadır. Bu testler kişiden alınan boğaz ve/veya burun sürüntüsünde virüsün genetik materyalinin aranması prensibine dayanır",
    "Bu hastalık için geliştirilmiş bir ilaç yoktur. Hastanın genel durumuna göre gerekli destek tedavisi uygulanmaktadır.",
    "Virüsler cansız ortamlarda kısa süre sonra ölmektedir. Bu nedenle mektup veya kargo paketlerinden COVID-19 bulaşması beklenmemektedir. Dünya Sağlık Örgütü, CDC ve ECDC’nin bu konuda kısıtlaması yoktur. Esas bulaş yolu virüsü taşıyan kişi ile yakın temastır. Kargoyu getiren kişiyle yakın temas edilmemeli, paket alındıktan sonra eller en az 20 saniye boyunca su ve sabunla yıkanmalıdır.",
    "Burnu tuzlu suyla yıkamanın COVID-19’dan korunmaya faydası olacağı düşünülmemektedir.",
    "Sirke kullanmanın COVID-19’dan korunmaya faydası olacağı düşünülmemektedir.",
    "Test sonucunuzu Bakanlığımızın E-nabız uygulaması üzerinden kolayca öğrenebilirsiniz. Eğer bilgisayarınız veya akıllı telefonunuz yoksa aile hekiminizden öğrenebilirsiniz.",
    "Koronavirüs testi sonucu pozitif çıkan bir kişiyle temas ettiyseniz karantina süreniz son temasınızdan bir gün sonra başlar ve 14 gün sürer. Korona testi sonucunuz pozitif ise,, Eğer semptomunuz yoksa veya hastalığınız hafif ise (evde tedavi aldıysanız veya hastanede 24 saatten kısa süre yattıysanız) test verdikten sonra izolasyon süreciniz başlar ve 10 gün süre ile izolasyonda kalırsınız., Eğer hastanede yatarak tedavi olduysanız (hastanede 24 saatten uzun süre yatmanız gerektiyse) test verdikten sonra izolasyon süreciniz başlar ve 14 gün süre ile izolasyonda kalırsınız., Eğer yoğun bakımda yatarak tedavi olduysanız veya bağışıklık sisteminizi baskılayan ek bir hastalığınız varsa test verdikten sonra izolasyon süreciniz başlar ve 20 gün süre ile izolasyonda kalırsınız. ",
    "İzolasyon ve karantina süreleri; hastalığa ait yeni bilimsel veriler, DSÖ, ECDC ve CDC’nin önerileri göz önünde bulundurularak düzenlenmektedir. Virüs, kişinin vücuduna girdikten sonra 14 gün içerisinde çoğalır ve hastalık yapar. Bu nedenle kişinin hastalık belirtileri açısından 14 gün boyunca karantinada kalması ve takip edilmesi hem hastalığın erken tespit edilmesi hem de diğer kişilere bulaşmasının önlenmesi için önemlidir. Ancak bir kişide hastalık çıktıktan sonra virüsün atılım süresi hastalığın şiddetine göre değişmektedir. Bu nedenle de bilimsel veriler göz önünde tutularak düzenlenen Koronavirüs hastasının izlem süreleri sabit olmayıp hastalığın şiddetine göre 10 gün, 14 gün veya 20 gün olarak değişmektedir.",
    "Hayır. Sağlık ihtiyaçlarınız için her zaman sağlık kuruluşlarına başvurabilirsiniz. Bu başvuru sırasında maskenizi takmalı, diğer kişilerle sosyal mesafeyi korumalı ve sizinle temas edecek sağlık çalışanlarına durumunuzu bildirmelisiniz. Sağlık kuruluşuna varsa özel aracınızla gitmeli, yoksa 112 acil sağlık ekiplerinden yardım almalısınız.",
    "İlaçlarınızı muhakkak hekiminizin veya ilacınızı size teslim eden sağlık görevlisinin önerdiği şekilde kullanmalısınız. Hekiminizin size farklı bir önerisi olmadıysa genel olarak hidroksiklorokin etken maddeli ilacı 5 gün boyunca sabah akşam birer tane; favipiravir etken maddeli ilacı ise ilk gün sabah sekiz, akşam sekiz tane, sonraki günlerde ise sabah üç, akşam üç tane olacak şekilde toplam 5 gün boyunca içmelisiniz.",
    "Aile hekiminiz belirlenen günlerde sizi arayarak genel sağlık durumunuzu soracak, sizi izleyecek, gerekirse il/ilçe sağlık yönetimi, hastane veya 112 acil sağlık ekipleri ile iletişim kurarak sağlık durumunuz hakkında onları bilgilendirecektir. Eğer hasta değil, temaslı iseniz sizde Koronavirüs hastalığı gelişip gelişmediğini anlamak amacıyla sorular soracak, gerekli hallerde sizi hastaneye yönlendirecektir.",
    "Bu durumda ikamet adresinize dönmeniz mümkün değildir. Hastalığınızın başka bir ilde tespit edilmesi durumunda sizden bulunduğunuz yerde kalmanız talep edilecek, o ildeki Vefa Destek Ekiplerinin önerdiği ya da belirlenmiş yurtlarda konaklamanız istenecektir.",
    "İzolasyon ya da karantina uygulamasında sağlık yönetiminin amacı diğer kişilerle olan teması engellemek ve bu yolla hastalığın yayılmasını durdurabilmektir. Bu nedenle adres değişikliği yapmak, temasa sebep olacağı için uygun bulunmamaktadır. Ancak yaşlı ya da çocuklar gibi bakıma muhtaç durumdaki kişiler için mümkün olan en az temasla ilçe sağlık yönetimleri ile görüştükten sonra adres değişikliği yapılmaktadır",
    "Hayır. Filyasyon çalışmaları sırasında sağlık personeli, testi pozitif çıkan kişi ile görüştükten sonra temaslıları belirlemektedir. Temaslı olarak ismi bildirilen bireylerle iletişim kurularak temasın doğruluğu onaylatılır. Daha sonra isim sisteme girilir ve sonrasında silinemez.",
    "Emzirmek, bebeğin büyümesi ve gelişmesi için önceliklidir. Bu nedenle annenin bebeğini emzirmeye devam etmesi önerilir. Ancak anne hasta olduğu dönemde, bebeğinin bakımını yaparken ve emzirirken bütün izolasyon kurallarına uymalıdır. Bebeğini emzirmeden önce ellerini yıkamalı, maskesini takmalı ve meme başı temizliği yapmalıdır. Sonrasında bebeğini güvenle emzirebilir. Eğer Koronavirüs nedeniyle ilaç kullanıyor iseniz bebeğinizi emzirmemeniz önerilmektedir. İlaç kullandığınız süre içinde sütünüzün sağınız ve dökünüz; ilaç tedaviniz bittikten sonra emzirmeye devam edebilirsiniz.",
    "İzolasyon süresi dolduktan sonra kişi bulaştırıcı kabul edilmez. Ancak, izolasyon süresi biten kişi, kurallar gereği evinin dışına çıktığı zamanlarda maske takmaya devam etmek zorundadır. Ayrıca sosyal mesafe kuralına uyulması sadece Koronavirüs’ten değil, birçok bulaşıcı hastalıktan korunmak için etkili bir yöntemdir.",
    "Hayır. Size önerilen izlem süresi bittikten sonra gündelik hayatınıza ve işinize dönebilirsiniz.",
    "Hastalığınız süresince hastanede yatarak tedavi edilirseniz sizi takip eden hekim, evinizde tedavi görürseniz aile hekimiz tarafından Bakanlığımızın E-rapor sistemi aracılığıyla sağlık raporunuz düzenlenebilir.",
    "Evde numune alımı, 60 yaşının üstündeki temaslı veya kronik hastalığı bulunan temaslılara uygulanmaktadır. Saha ekipleri size ilk geldiğinde semptomlarınız var ise sizlerden numune alınmaktadır.",
    "Evet. Sınava girmeden önce sınavı yapacak kurumun internet sayfasını kontrol ederek Koronavirüs hakkındaki bilgileri okuyunuz. Sınava giderken maskenizi takın ve sınav girişinde sınav görevlilerine hasta ya da temaslı olduğunuzu bildirin."
]

questions = ["<Q>" + q for q in questions]
answers = ["<A>" + a for a in answers]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name)


def answer_faq(model, tokenizer, questions, answers, return_similarities=False):
    q_len = len(questions)
    tokens = tokenizer(questions + answers, padding=True, return_tensors='tf')
    embs = model(**tokens)[0]

    attention_masks = tf.cast(tokens['attention_mask'], tf.float32)
    sample_length = tf.reduce_sum(attention_masks, axis=-1, keepdims=True)
    masked_embs = embs * tf.expand_dims(attention_masks, axis=-1)
    masked_embs = tf.reduce_sum(masked_embs, axis=1) / tf.cast(sample_length, tf.float32)
    a = tf.math.l2_normalize(masked_embs[:q_len, :], axis=1)
    b = tf.math.l2_normalize(masked_embs[q_len:, :], axis=1)

    similarities = tf.matmul(a, b, transpose_b=True)

    scores = tf.nn.softmax(similarities)
    results = list(zip(answers, scores.numpy().squeeze().tolist()))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_results = [{"answer": answer.replace("<A>", ""), "score": f"{score:.4f}"} for answer, score in
                      sorted_results]
    return sorted_results


for question in questions:
    results = answer_faq(model, tokenizer, [question], answers)
    print(question.replace("<Q>", ""))
    print(results[0])
    print("---------------------")
