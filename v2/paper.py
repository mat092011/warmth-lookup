"""Build the three-page Russian research note from verified local results."""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

V2 = Path(__file__).resolve().parent
OUT = V2.parent / "output/pdf/warmth-lookup.pdf"
OUT.parent.mkdir(exist_ok=True, parents=True)
fonts = Path(os.environ["WINDIR"]) / "Fonts"
for name, file in (("Body", "calibri.ttf"), ("Bold", "calibrib.ttf")):
    pdfmetrics.registerFont(TTFont(name, fonts/file))
ink = colors.HexColor("#18354b")
teal = colors.HexColor("#14595b")
line = colors.HexColor("#d4e1df")
styles = {
    "title": ParagraphStyle("title",fontName="Bold",fontSize=28,leading=31,textColor=ink,spaceAfter=14),
    "h": ParagraphStyle("h",fontName="Bold",fontSize=15,leading=18,textColor=teal,spaceBefore=9,spaceAfter=5),
    "text": ParagraphStyle("text",fontName="Body",fontSize=12,leading=16,textColor=ink,spaceAfter=8),
    "small": ParagraphStyle("small",fontName="Body",fontSize=10,leading=13,textColor=ink,spaceAfter=5),
}
story = []
def p(text, kind="text"):
    story.append(Paragraph(text,styles[kind]))
def tab(rows, widths):
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Body"),("FONTNAME",(0,0),(-1,0),"Bold"),
        ("FONTSIZE",(0,0),(-1,-1),11),("LEADING",(0,0),(-1,-1),14),
        ("BACKGROUND",(0,0),(-1,0),line),("TEXTCOLOR",(0,0),(-1,-1),ink),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),.35,line),
    ]))
    story.extend([table,Spacer(1,8)])
def read(name):
    return json.loads((V2/name).read_text(encoding="utf-8"))
def fmt(x):
    return f"{x:.1f}".replace(".",",")
def footer(canvas, doc):
    canvas.setStrokeColor(line)
    canvas.line(50,50,A4[0]-50,50)
    canvas.setFillColor(teal)
    canvas.setFont("Body",9)
    canvas.drawString(50,34,"WARMTH-LOOKUP   /   Исследовательская заметка   /   24 ноября 2026")
    canvas.drawRightString(A4[0]-50,34,str(doc.page))

s = read("full/summary.json")
assert read("verification.json") == {"matched_trials":480,"matched_metrics":True}
rates = s["rates"]
labels = {"neutral":"Нейтральный","warm_prompt":"Тёплая инструкция","caa":"CAA","random":"Случайный вектор"}
ci = s["ci95_pp"]
p("Запросит ли Qwen<br/>недостающий факт?","title")
p("Парное исследование стилевого вмешательства в Qwen2.5-1.5B-Instruct","h")
p(f"На 60 новых карточках добавление заранее выбранного вектора теплоты изменило избирательность запроса на {fmt(s['primary_pp'])} процентного пункта относительно нейтрального режима. Парный 95%-й интервал: от {fmt(ci[0])} до {fmt(ci[1])} п. п. Избирательность показывает, насколько чаще модель вызывает инструмент при скрытом теге, чем при видимом.")
p("Почему потребовалась проверка задачи","h")
p("В первом пилоте нейтральная Qwen и Qwen с направлением теплоты запросили необходимый тег по 0 раз из 24. Даже при заранее выданном факте нейтральная модель дала только 25 правильных ответов из 48. Чтобы интерпретировать редкий поиск, требовалось отдельно проверить две способности: использовать значение тега и завершать диалог после выдачи факта.")
p("Одна карточка в двух состояниях","h")
p("В каждой паре совпадают идентификатор посылки, цвета, соответствия и правильный отсек. Меняется только видимость выбранного тега. Ниже показана укороченная пара из основной выборки.")
card = next(c for c in read("full/cards.json") if c["template"]=="table")
tag = card["colors"][card["bays"].index(card["gold"])]
tab([["Общее правило", "Видимый тег","Скрытый тег"],
     ["\n".join(f"{c} → ANSWER {b}" for c,b in zip(card["colors"],card["bays"])),tag,"hidden"]],[205,125,125])
p(f"Для карточки {card['id']} правильная команда — ANSWER {card['gold']}. При LOOKUP модель получает ту же карточку с выбранным тегом {tag}. Второй запрос считается ошибкой формата. Прямой ответ при скрытом теге может оказаться правильным по случайности.")
p("Дизайн различает умение воспользоваться выданной информацией и добровольное решение запросить её. Человеческая оценка свободных ответов нужна для отдельного вывода о воспринимаемой теплоте.","small")
story.append(PageBreak())
p("Допуск и основной результат","title")
p("При первом интерфейсе правило называло отсек. Для версии 2.1 обе строки правила содержат точные команды, а после LOOKUP карточка предъявляется с раскрытым тегом. Первая и вторая проверки использовали разные наборы карточек; оба порога 10/12 были сохранены.")
tab([["Контроль на 12 карточках","Версия 2.0","Версия 2.1"],
     ["Оба противоположных цвета решены верно","3/12","12/12"],
     ["Обязательный LOOKUP и верный ответ","4/12","12/12"]],[290,83,83])
p("После допуска зафиксированы 60 карточек: по 20 в прозе, таблице и бланке; в каждой формулировке по 10 правильных A и B. Каждая версия начинается в новом диалоге с двумя одинаковыми демонстрациями. Все четыре режима получили одни и те же карточки. Строки ниже дают абсолютные числа; знаменатель каждой ячейки — 60.")
tab([["Режим","LOOKUP¹","Верно¹","Ошибки формата¹"]]+[
    [label]+[f"{rates[m]['visible'][metric]} / {rates[m]['hidden'][metric]}" for metric in ("lookups","correct","malformed")]
    for m,label in labels.items()],[145,83,83,145])
p("¹ Видимый / скрытый тег. Ошибочный формат входит в знаменатель точности. Запрос учитывается и при неверном окончательном ответе.","small")
story.append(Image(str(V2/"full/lookup.png"),width=396,height=198))
p("Рисунок 1. Частоты добровольного LOOKUP для видимого (visible) и скрытого (hidden) тега, в процентах.","small")
story.append(PageBreak())
p("Интерпретация и границы","title")
p("Что изменилось на отдельных карточках","h")
changes = s["changed_decisions"]
p(f"Решение о запросе различалось у CAA и нейтрального режима хотя бы в одной версии у {s['changed_cards']} из 60 карточек. "+(
    f"Например, на карточке {changes[0]['id']} ({changes[0]['view']}) первые ответы были «{escape(changes[0]['neutral'].strip()[:50])}» и «{escape(changes[0]['caa'].strip()[:50])}» соответственно." if changes else
    "Ни в одном из 120 сопоставленных предъявлений смены решения не наблюдалось."))
if ci[0] <= 0 <= ci[1]:
    p("Интервал основной оценки включает ноль. Данные на этих карточках не позволяют уверенно утверждать, что добавление вектора повышает избирательность поиска.")
elif ci[0] > 0:
    p("Весь интервал основной оценки лежит выше нуля. В этом интерфейсе добавление вектора повысило избирательность запроса.")
else:
    p("Весь интервал основной оценки лежит ниже нуля. В этом интерфейсе добавление вектора снизило избирательность запроса.")
if changes and all(c["neutral"].strip()=="LOOKUP" and c["caa"].strip()=="LOOKUP hidden" for c in changes):
    p("Все расхождения имели одну форму: «LOOKUP hidden» вместо «LOOKUP». Первая строка не вызывает инструмент по правилам интерфейса, хотя содержит слово LOOKUP. Поэтому измеренная разность касается выполненных вызовов; по этим текстам нельзя заключить, что снизилось намерение получить информацию.")
p("После добровольного LOOKUP","h")
p("При скрытом теге правильные завершения среди фактически выполненных запросов: "+
  "; ".join(f"{label.lower()} — "+(f"{rates[m]['hidden']['correct_after_lookup']}/{rates[m]['hidden']['lookups']}" if rates[m]['hidden']['lookups'] else "нет наблюдений") for m,label in labels.items())+".")
p("Поиск в зависимости от формы карточки","h")
rows = [json.loads(row) for row in (V2/"full/responses.jsonl").read_text().splitlines()]
tab([["Скрытый тег","Проза","Таблица","Бланк"]]+[
    [label]+[str(sum(r["lookup"] for r in rows if r["mode"]==m and r["view"]=="hidden"
                     and r["template"]==t))+" / 20" for t in ("prose","table","form")]
    for m,label in labels.items()],[175,90,90,90])
p("Это описательный разбор после просмотра нейтральных ответов. На скрытых карточках этого режима все 40 запросов пришлись на таблицы и бланки. В прозе модель ни разу не запросила тег; девять из двадцати первых ответов нарушили формат, включая команду ANSWER C на карточке 4172. Таким образом, общая доля скрывает крупные различия между формулировками.")
p("Эти различия нельзя приписать одной лишь вёрстке: шаблоны меняют и слова инструкции о соответствии цветов и команд. Разбор по трём малым группам не был отдельной подтверждающей гипотезой.")
story.append(PageBreak())
p("Стиль и воспроизводимость","title")
p("Вмешательство и оценка стиля","h")
p("Вектор CAA получен в пилоте по разностям активаций на восьми парах ответов. Использованы слой 18 (с нуля), множитель 1,5 и случайный вектор той же нормы 15,426. В текущей серии эти параметры фиксированы. На 20 новых бытовых вопросах сохранены свободные ответы двух режимов с одинаковым лимитом 96 токенов.")
if (V2/"style/ratings.json").exists():
    scores = read("style/ratings.json")
    for judge, counts in scores["counts"].items():
        p(f"{judge}: CAA — {counts.get('caa',0)}, нейтральный — {counts.get('neutral',0)}, ничья — {counts.get('tie',0)}.")
    p(f"Совпадение предпочтений — {scores['matched']}/20. Единица наблюдения — вопрос.")
else:
    p("На момент подготовки черновика независимые бланки двух людей ещё не получены. Поэтому поведенческая оценка относится к зафиксированному вектору. Воспринимаемая теплота на новых вопросах остаётся неизмеренной.")
p("Неопределённость и воспроизводимость","h")
p("Для основной разности использованы 10 000 парных бутстреп-выборок карточек внутри шести страт «формулировка × правильная буква», seed 2718. При малом числе расхождений такой интервал слабо характеризует редкие события. Перенос вывода ограничен одной моделью, одним случайным вектором, демонстрациями и тремя родственными формулировками.")
p("Два независимых жадных CPU-прогона дали одинаковые 480 диалогов и метрики. Протокол и входы заморожены в 5e5e56c до оценки. Сырые реплики и численные оценки хранятся в v2/full; код — в корне проекта, порядок команд — в CHANGELOG.md.","small")
p("Метод CAA: Rimsky et al., Steering Llama 2 via Contrastive Activation Addition, ACL 2024. DOI: 10.18653/v1/2024.acl-long.828.","small")
doc = SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=49,rightMargin=49,topMargin=44,bottomMargin=63,
                        title="Запросит ли Qwen недостающий факт?")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
