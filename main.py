import json
import random
import nltk
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser

# Nämä vaadittiin nltk:n toimintaan, punkt_tab oli vielä pakko lisätä uudemman pythonin vuoksi
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Tehdään PerunaRobosta luokka joka käyttää hyväkseen json tiedostoa jossa on keskusteluvaihtoehdot
class PerunaRobo:
    def __init__(self, json_path):
        with open(json_path, encoding='utf-8') as f:
            self.intents = json.load(f)

        self.all_patterns = []
        self.pattern_to_intent = []

        # käy läpi json tiedostosta löytyviä mahdollisia keskusteluita ja niiden vastauksia, kunnes löytää parhaimman vaihtoehdon
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tallennetaan alkuperäinen teksti pienellä kirjoitettuna
                self.all_patterns.append(pattern.lower())
                self.pattern_to_intent.append(intent)

        # Käytetään char-tason analysaattoria (n-grams 2-5 merkkiä)
        # Lemmaaminen ei toiminut suomen kielellä oikeastaan ollenkaan taivutusmuotojen ja puhekielen takia
        # Tämä taas toimi äärettömän hyvin
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.all_patterns)

    def get_response(self, user_input):
        user_input = user_input.lower() # muutetaan käyttäjän input pieniksi kirjaimiksi jotta niiden prosessointi helpottuu

        # Muutetaan käyttäjän syöte vektoreiksi
        user_tfidf = self.vectorizer.transform([user_input])

        # Lasketaan samankaltaisuus, tämä löytyi suoraa linkin takaa ja käytin tekoälyä selittääkseni asiaa itselleni paremmin, en vieläkään
        # ymmärrä täysin mitä taustalla tapahtuu mutta jollain matemaattisella kaavalla tässä lasketaan todennäköisyyksiä ja tehdään vertailua
        similarity_scores = cosine_similarity(user_tfidf, self.tfidf_matrix)
        closest_match_index = similarity_scores.argsort()[0][-1]
        max_score = similarity_scores[0][closest_match_index]

        # Kynnysarvo oli hyvä säätää pieneksi, silloin vertailu toimii oikein ja botti ei anna virhettä vaikka se tunnistaisi sanan
        if max_score < 0.12:
            return "Pahoittelut, en ymmärtänyt kysymystäsi täysin. Voisitko kokeilla eri sanoilla, kuten 'yhteystiedot' tai 'hyväksiluku'?"

        matched_intent = self.pattern_to_intent[closest_match_index]
        return random.choice(matched_intent['responses'])

# Luodaan luokka itse chat ikkunalle
class ChatInterface:
    def __init__(self, root, bot):
        self.root = root
        self.bot = bot
        self.root.title("PerunaRobo")
        self.root.geometry("500x600")

        # Otsikko
        tk.Label(root, text="PerunaRobo valmiina auttamaan", bg="#005b96", fg="white", font=("Arial", 12, "bold"), pady=10).pack(fill=tk.X)

        # Chat-ikkuna ja sen muotoilu. Tässä luodaan tägit jo käyttäjälle ja botille että viestit menevät oikeaan osoitteeseen (vasen ja oikea)
        self.chat_window = scrolledtext.ScrolledText(root, bd=0, bg="white", font=("Arial", 11), state=tk.DISABLED, wrap=tk.WORD) # Wrap auttaa sanojen katkeamiseen
        self.chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_window.tag_configure("kayttaja", justify='right', foreground="#005b96")
        self.chat_window.tag_configure("botti", justify='left', foreground="#333333")

        # Lisää uusi tägi linkeille
        self.chat_window.tag_configure("linkki", foreground="blue", underline=True)
        # Muutetaan hiiri kädeksi, kun se on linkin päällä
        self.chat_window.tag_bind("linkki", "<Enter>", lambda e: self.chat_window.config(cursor="hand2"))
        self.chat_window.tag_bind("linkki", "<Leave>", lambda e: self.chat_window.config(cursor=""))

        # Alue syötteelle
        self.entry_frame = tk.Frame(root)
        self.entry_frame.pack(fill=tk.X, padx=10, pady=10)
        # Kirjoituskenttä viestille
        self.user_entry = tk.Entry(self.entry_frame, font=("Arial", 12))
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.user_entry.bind("<Return>", lambda e: self.send_message())
        # Nappi johon on sidottu lähetysfunktio
        self.send_btn = tk.Button(self.entry_frame, text="Lähetä", command=self.send_message, bg="#005b96", fg="white")
        self.send_btn.pack(side=tk.RIGHT, padx=5)

        self._display_msg("PerunaRobo", "Terve! Olen PerunaRobo! Autan sinua perunoiden kasvatuksessa!")

    # funktio viestien näyttämiseen
    def _display_msg(self, sender, text):
        self.chat_window.config(state=tk.NORMAL)

        # Määritetään viestin paikka ikkunassa, botti menee vasemmalle ja käyttäjä oikealle
        tag = "kayttaja" if sender == "Sinä" else "botti"

        # Lisätään ensin lähettäjän nimi omalle rivilleen
        self.chat_window.insert(tk.END, f"{sender}:\n", tag)

        # Halusin lisätä vielä linkkien tunnistuksen, tämä oli näppärin keino minkä löysin asiaan. Tekstistä poistetaan välilyönnit ja katsotaan
        # onko joku siinä olevista sanoista linkki. Tähän tuli vinkki tekoälyltä jota sitten tutkin ja opiskelin
        words = text.split(" ")
        for i, word in enumerate(words):
            if word.startswith("http://") or word.startswith("https://"): # tarkistetaan alkaako sana näillä kahdella tavalla, tässä oli tärkeää että intents tiedostossa linkit oli oikein muotoiltu ja että ne alkavat näillä
                # Luodaan uniikki tägi tälle nimenomaiselle linkille
                link_id = f"link-{random.randint(0, 99999)}"
                # Lisätään teksti ja annetaan sille sininen tyylittely ja sille luotu uniikki ID
                self.chat_window.insert(tk.END, word, ("linkki", link_id))
                # Sidotaan klikkaus avaamaan linkki selaimessa, ja nimenomaan hiiren 1 painike
                self.chat_window.tag_bind(link_id, "<Button-1>", lambda e, url=word: webbrowser.open(url))
            else:
                # Jos sana ei ole linkki, lisätään se tavallisena tekstinä viestin tägillä
                self.chat_window.insert(tk.END, word, tag)

            # Käydään lause loppuun ja lisätään poistetut välilyönnit takaisin.
            if i < len(words) - 1:
                self.chat_window.insert(tk.END, " ")

        # Lisätään loppuun rivinvaihdot ja rullataan alas, näin näkymä pysyy siistinä ja ruudulla on aina ajantasaisin viesti
        self.chat_window.insert(tk.END, "\n\n")
        self.chat_window.see(tk.END)
        self.chat_window.config(state=tk.DISABLED)

    # funktio viestien lähetykseen ja näkymiseen teksti-ikkunassa
    def send_message(self):
        msg = self.user_entry.get().strip()
        if msg:
            self.user_entry.delete(0, tk.END) #jos viesti lähetetään, tyhjennetään viestikenttä tekstistä
            self._display_msg("Sinä", msg) # näytetään viesti ikkunassa käyttäjän tägillä
            res = self.bot.get_response(msg) # botti hakee vastauksen intents tiedostosta
            self._display_msg("PerunaRobo", res) # näytetään viestin lähettäjänä OpoRobo ja botin hakema viesti

# Ajetaan ohjelma ja kutsutaan luotuja luokkia
if __name__ == "__main__":
    bot = PerunaRobo('intents.json')
    root = tk.Tk()
    ChatInterface(root, bot)
    root.mainloop()