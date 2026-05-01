import random as r
import math as m
import datetime
from .savegl import save_game, load_game

ln = m.log
rfloat = r.uniform 
def twofloat(arg): return format(arg, '.2f')

class CityClass():
    
    def __init__(self,name):
        self.name = name
        self.inttur = rfloat(0,2)
        self.ano_fundacao = int(datetime.date.today().year)
        
        # Placeholders
        self.idh = 0.750
        self.exvida = 75
        self.fluxoesc = 0.6
        self.renda = 1200
        self.doenca_escolhida = "Rubéola"
        
        # Valores Iniciais
        self.classificacao = "Vila"
        
        self.t = 0
        self.pop = 1000
        self.taxresid = 0.15
        self.taxrural = 0.15
        self.taxind = 0.15
        self.taxcomer = 0.15

        self.fatrural = 0.9
        self.fatind = 0.05
        self.fatcomer = 0.05

        self.invsaude = 100000 
        self.invesc = 100000
        self.invcomer = 100000
        self.invtur = 100000
        self.invref = 100000
        self.invpesq = 100000
        self.invtotal = self.invsaude + self.invesc + self.invtur + self.invcomer + self.invref + self.invpesq

        self.epid = 0
        self.escadulto = 0.7

        self.escola = []
        self.popul = []
        self.epid_last = 0

        self.fat = ((self.fatrural * rfloat(0,2)) + (self.fatind * rfloat((1/2),(3/2)) * (3/2)) + (self.fatcomer * rfloat(0,(3/2)) * 2)) * self.renda * self.pop
        self.imptotal = (self.renda * self.taxresid * self.pop) + ((self.fatcomer * self.taxcomer) + (self.fatrural * self.taxrural) + (self.fatind * self.taxind) * self.fat)
        
        self.cofre = 1000000 + self.imptotal
        
        self.update_variaveis()

    
    def update_variaveis(self):
        # Lista de variáveis e seus valores (usados ao salvar ou carregar um jogo)
        s = self
        s.variaveis = {
            "name": s.name, "inttur": s.inttur, 
            "ano_fundacao": s.ano_fundacao, "idh": s.idh, "exvida": s.exvida,
            "fluxoesc": s.fluxoesc, "renda": s.renda,
            "doenca_escolhida": s.doenca_escolhida, "classificacao": s.classificacao,
            "t": s.t, "pop": s.pop, "taxresid": s.taxresid, "taxrural": s.taxrural,
            "taxind": s.taxind, "taxcomer": s.taxcomer, "fatrural": s.fatrural,
            "fatind": s.fatind, "fatcomer": s.fatcomer, "invsaude": s.invsaude,
            "invesc": s.invesc, "invtur": s.invtur, "invcomer": s.invcomer,
            "invref": s.invref, "invpesq": s.invpesq, "invtotal": s.invtotal,
            "epid": s.epid, "escadulto": s.escadulto, "escola": s.escola, 
            "popul": s.popul, "epid_last": s.epid_last, "fat": s.fat, 
            "imptotal": s.imptotal, "cofre": s.cofre
        }
    
    
    def next(self):    
        s = self
        
        """Eventos aleatorios"""
        
        # Correção de erros matemáticos (como ln(0))
        correct = [s.invsaude, s.invesc, s.invtur, s.invcomer, s.invref, s.invpesq, s.pop]
        for num in correct:
            if num < 1:
                num = 1
                
        taxs = [s.taxresid, s.taxind, s.taxcomer]
        for tax in taxs:
            if tax < 0.005:
                tax = 0.005
        
        # Chance de epidemia
        currentepid = s.epid
        epidch = rfloat(0,1)
        if currentepid > 0:
            if epidch < 0.8: s.epid = (- ln(ln(s.invsaude) / 20)) * (ln(s.pop) / 10) * rfloat(1/2 , 3/2)
            if epidch >= 0.8: s.epid = 0
        if currentepid == 0:
            if epidch >= 0.9: 
                s.epid = (- ln((ln(s.invsaude) / 20))) * (ln(s.pop) / 10) * rfloat(1/2 , 3/2)
                doencas = ["Influenza", "Sarampo", "Chikungunya", "Coqueluche", "Febre Zika", "Dengue", 
                "Tuberculose", "Leishmaniose", "Rubéola"] 
                s.doenca_escolhida = r.choice(doencas)
            if epidch < 0.9: s.epid = 0
        
        s.epid_last = currentepid
        
        # Chance de ...

        # Registro do Historico de Invesc e Pop
        s.escola.insert(s.t, s.invesc) 
        s.popul.insert(s.t, s.pop)

        # Atividades economicas
        crescind = m.sqrt(m.sqrt((m.log(s.pop,10)) * m.log(s.invpesq,10)) * -(m.log(s.taxind,10))) * ((3/2) ** s.fatrural)
        s.fatind *= crescind
        cresccomer = m.sqrt((m.log(s.invtur,10)) * (m.log(s.invcomer,10)) * s.inttur) * -(m.log(s.taxcomer,10) / 2) * ((3/2) ** s.fatind)
        s.fatcomer *= cresccomer
        if (s.fatind + s.fatcomer) <= 1: s.fatrural = 1 - (s.fatind + s.fatcomer)
        else: s.fatrural = (m.ceil(s.fatind + s.fatcomer) - (s.fatind + s.fatcomer)) * 2
        if s.fatrural + s.fatcomer + s.fatind != 1:
            desfat = s.fatrural + s.fatcomer + s.fatind
            s.fatrural /= desfat
            s.fatcomer /= desfat
            s.fatind /= desfat

        # Saude
        s.exvida = (ln(s.invsaude - 5000) * 30 / ln(s.pop ** (2))) + 45 - (3 * s.epid) - (3.5 * s.fatind) + (1.5 * s.fatcomer)
        """DEFINIR SAUDE POR ATV ECONOMICA"""
        if s.exvida > 85: s.exvida = 85

        # Fluxo escolar
        s.fluxoesc = (ln(s.invesc) / ln(s.pop * 10000))
        if s.fluxoesc > 1: s.fluxoesc = 1
    
        # Escolaridade adulta
        if s.t >= 6: s.escadulto = (ln(s.escola[s.t-6]) / ln(s.popul[s.t-6] * 10000))
        else: s.escadulto = 0.8
        if s.escadulto > 1: s.escadulto = 1

        # Indices e IDH
        s.invida = (s.exvida - 25) / (85 - 25)
        s.inedu = ((s.fluxoesc ** 2) * s.escadulto) ** (1 / 3)
        
        # Calculo de Renda Media              Fator rural                  Fator Industria                   Fator Comercio              
        s.renda = s.inedu * 4033 * ((ln(s.invref) * s.fatrural / ln(s.pop ** 4)) + (s.fatind * s.escadulto) + (rfloat((1/2),(3/2)) * s.inttur * s.fatcomer / 3))
        if s.renda > 4033: s.renda = 4033
    
        # Indices e IDH 2
        s.inrenda = (ln(s.renda) - ln(8)) / (ln(4033) - ln(8))
        s.idh = (s.invida * s.inedu * s.inrenda) ** (1/3)
        
        # Impostos e Faturamento (Aqui se calcula a Safra, Mercado e Temporada)
        s.fat = ((s.fatrural * rfloat(0,2)) + (s.fatind * rfloat((1/2),(3/2)) * (3/2)) + (s.fatcomer * rfloat(0,(3/2)) * 2)) * s.renda * s.pop
        s.imptotal = (s.renda * s.taxresid * s.pop) + ((s.fatcomer * s.taxcomer) + (s.fatrural * s.taxrural) + (s.fatind * s.taxind)) * s.fat
    
        # Crescimento Populacional
        rate = ln(s.inedu, (1/2)) + (((4 * s.idh) - 2) * s.inttur) - ((ln(1 / s.exvida) - ln(1 / 100)) * 3) - (m.log(s.taxresid,10) + 1.3)
        s.pop = m.ceil(s.pop * (m.exp(rate * rfloat(0,2))))

        # Cofre
        s.invtotal = s.invsaude + s.invesc + s.invtur + s.invcomer + s.invref + s.invpesq
        s.cofre = s.cofre - s.invtotal + s.imptotal
        
        s.t += 1
        
        # Definição da classificação da cidade
        if (s.pop <= 50000):
            s.classificacao = "Vila"
        elif (s.pop > 50000) and (s.pop <= 100000):
            s.classificacao = "Município"
        elif (s.pop > 100000) and (s.pop <= 900000):
            s.classificacao = "Cidade"
        elif (s.pop > 900000) and (s.pop <= 10000000):
            s.classificacao = "Metrópole"
        elif (s.pop > 10000000):
            s.classificacao = "Megalópole"
        
        s.update_variaveis()
        save_game(self)
        
        
    def noticias(self):
        s = self
        
        # Definição de temas possíveis para as notícias
        themes = []
        if s.epid > 0 and s.epid_last == 0: themes.append("epidstrt")
        if s.epid > 0 and s.epid_last > 0: themes.append("epidcont")
        if s.epid == 0 and s.epid_last > 0: themes.append("epidend")
        
        if s.fluxoesc < 0.5: themes.append("fluxoescpior")
        if s.fluxoesc > 0.95: themes.append("fluxoescmelhor")
        
        if s.exvida < 60: themes.append("exvidapior")
        if s.exvida > 80: themes.append("exvidamelhor")
        
        if s.renda < 100: themes.append("rendapior")
        if s.renda > 3500: themes.append("rendamelhor")
        
        # Escolha de Temas
        news = {"epidstrt": ("Casos confirmados de {} aumentam; Ministério da Saúde confirma estado de epidemia".format(s.doenca_escolhida)), 
            "epidcont":("Epidemia continua: Mais casos de {} são confirmados".format(s.doenca_escolhida)),
            "epidend":("Epidemia de {} é controlada com sucesso; Habitantes aliviados".format(s.doenca_escolhida)),
            "fluxoescpior":"Com {}%, o fluxo escolar de {} é considerado um dos piores do país".format(twofloat(s.fluxoesc * 100), s.name),
            "fluxoescmelhor":"Fluxo escolar de {}% leva a cidade de {} a uma das lideranças em educação do país!".format(twofloat(s.fluxoesc * 100), s.name),
            "exvidapior":"Com uma Expectativa de Vida de {} anos, habitantes de {} temem morte precoce".format(int(s.exvida), s.name),
            "exvidamelhor":"{}, cidade com uma Expectativa de Vida de {}, pode ter o segredo para uma vida longa".format(s.name, int(s.exvida)),
            "rendapior": "Habitantes de {} são uns dos que mais sofrem com a miséria no país, com uma renda média de {}".format(s.name, int(s.renda)),
            "rendamelhor": "A cidade de {} é nomeada como uma das cidades com menor índice de miséria e desemprego!".format(s.name)}
        """IMPORTANTE CONTINUAR"""
        
        for i in range(3):
            if themes == []:
                break
            tema_escolhido = r.choice(themes)
            print("  - " + news[tema_escolhido])
            themes.remove(tema_escolhido)
            
            