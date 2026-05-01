from telegram.ext import *
from telegram import *
import json
from datetime import datetime as dt
from CheckFunctions import *
import flask

updater = Updater('1676363369:AAHYqAQ2MxbytBdBPJkvtAxx1FwqBSzoXKw', use_context=True)
bot = Bot(token="1676363369:AAHYqAQ2MxbytBdBPJkvtAxx1FwqBSzoXKw")
dp = updater.dispatcher
group = None


# app = flask.Flask(__name__)


def main():
    # app.run(host="127.0.0.1", port=8080)
    updater.start_polling()
    updater.idle()


def check_codes(update, context):
    try:
        word = update.message.text.lower()
        with open("info/pupils.json", 'r', encoding="UTF-8") as pupils:
            dct = json.load(pupils)
        if update.message.from_user.id not in dct["groups"][dct["code_words"][word]]:
            dct["groups"][dct["code_words"][word]].append(update.message.from_user.id)
        pupils.close()
        update.message.reply_text(
            f'Ученик с ником{update.message.from_user.first_name} добавлен в группу {dct["groups"][dct["code_words"]]}')
        with open("info/pupils.json", 'w', encoding="UTF-8") as pupils:
            json.dump(dct, pupils, ensure_ascii=False)
        pupils.close()
    except Exception:
        return None


def set_timetable(update, context):
    if check_teacher(update.message.from_user.id):
        with open('info/timetable.json', 'r', encoding='UTF-8') as file:
            data = json.load(file)
            day = context.args[0]
        timetable = context.args[1:]
        data['week'][day] = timetable
        # Что если timetable - это какая-то чушь?
        file.close()
        with open('info/timetable.json', 'w', encoding='UTF-8') as file:
            json.dump(data, file, ensure_ascii=False)
        file.close()
        update.message.reply_text('Расписание обновлено')
    else:
        update.message.reply_text('Отказано в доступе')
    # ===================Log================================================================
    with open('info/log.txt', 'a', encoding='UTF-8') as history:
        history.write(f"{update.message.from_user.first_name} изменил расписание.")
    history.close()
    check_join(update.message.from_user.id)


def timetable(update, context):
    with open('info/timetable.json', 'r', encoding='UTF-8') as file:
        data = json.load(file)
        day = context.args[0]
    s = ''
    for i in data['week'][day]:
        s += i + '\n'
    file.close()
    update.message.reply_text(s)
    check_join(update.message.from_user.id)


def bell(update, context):
    with open('info/timetable.json', 'r', encoding='UTF-8') as file:
        data = json.load(file)
    s = ''
    for i in data['bell'].keys():
        s += f"{i} | {data['bell'][i]}\n"
    file.close()
    update.message.reply_text(s)
    check_join(update.message.from_user.id)


def add_teacher(update, context):
    password = context.args[0]
    name = " ".join(context.args[1:])
    if hash(password) == 1422319535244306737:
        with open('info/teachers.txt', 'a', encoding='UTF-8') as teachers:
            teachers.write(f'{name} {update.message.from_user.id}\n')
        update.message.reply_text(f'Учитель {name} добавлен.')
    else:
        update.message.reply_text('Ты не учитель, пшел вон отсюда')
    check_join(update.message.from_user.id)


def send_messages(update, context):
    with open("info/pupils.json", 'r', encoding="UTF-8") as pupils:
        dct = json.load(pupils)
    pupils.close()
    reply_keyboard = [dct["groups"].keys()]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text(
        'Участникам какой группы вы хотите отправить сообщение?',
        reply_markup=markup
    )
    return 1


def first_response(update, context):
    global group
    group = update.message.text
    update.message.reply_text("Отлично. Какое сообщение вы хотите отправить?")
    return 2


def second_response(update, context):
    global group
    msg = update.message.text
    name = ''
    id = update.message.from_user.id
    for s in open('info/teachers.txt', encoding='UTF-8').readlines():
        if int(s.strip().split()[-1]) == update.message.from_user.id:
            name = ''.join(s[:-10])
    message = name + ': ' + msg
    with open('info/pupils.json', 'r', encoding="UTF-8") as pupils:
        data = json.load(pupils)
        dont_send = open("info/doNotSend.txt", "r", encoding="UTF-8").readlines()
        print(dont_send)
        print(group)
        for i in data["groups"][group]:
            if (str(i) + '\n') not in dont_send:
                print(str(i) + '\n')
                bot.send_message(text=message, chat_id=i)
                print("отправлено")
        pupils.close()
    return ConversationHandler.END


def stop(update, context):
    return ConversationHandler.END


def add_marks(update, context):
    if check_teacher(update.message.from_user.id):
        try:
            marks = json.load(open('info/marks.json', 'r', encoding='UTF-8'))
        except Exception:
            marks = {}
        data = context.args
        pupil_name, half, subject, list_of_marks = data[0], data[1], data[2], data[3:]
        if pupil_name not in marks:
            marks[pupil_name] = {}
            marks[pupil_name]['1'] = {}
            marks[pupil_name]['2'] = {}
        if subject not in marks[pupil_name][half]:
            marks[pupil_name][half][subject] = []
        marks[pupil_name][half][subject] += list_of_marks
        json.dump(marks, open('info/marks.json', 'w', encoding='UTF-8'), ensure_ascii=False)
    else:
        update.message.reply_text('Функция предназначена только для учителей.')


def marks(update, context):
    access = []
    entered_name = context.args[0]
    half = context.args[1]
    answ = ''
    with open('info/marks.json', 'r', encoding="UTF-8") as marks_file:
        marks = json.load(marks_file)
    for teacher in open("info/teachers.txt", "r", encoding="UTF-8").readlines():
        access.append(int(teacher.strip().split()[-1]))
    for pupil in open("info/pupils.txt", "r", encoding="UTF-8").readlines():
        if entered_name == ' '.join(pupil.strip().split()[:-1]):
            access.append(int(pupil.strip().split()[-1]))
    if update.message.from_user.id in access:
        try:
            subjects = marks[entered_name][half]
        except Exception:
            update.message.reply_text("Нет такого ученика.")
            return True
        for subject in subjects.keys():
            answ += f"{subject}: {' '.join(marks[entered_name][half][subject])} ({average(marks[entered_name][half][subject])})\n"
        if not answ:
            update.message.reply_text("Нет оценок в этой четверти")
        update.message.reply_text(answ)
    else:
        update.message.reply_text("Отказано в доступе.")
        marks_file.close()


def mute(update, context):
    print("записан")
    with open("info/doNotSend.txt", "a", encoding="UTF-8") as file:
        file.write(str(update.message.from_user.id) + '\n')
        print("записан")
    file.close()


def add_group(update, context):
    if check_teacher(update.message.from_user.id):
        group_name = ' '.join(context.args[:-1])
        code = context.args[-1]
        with open("info/pupils.json", "r", encoding="UTF-8") as pupils:
            dct = json.load(pupils)
        if group_name not in dct["groups"]:
            dct["groups"][group_name] = []
            dct["code_words"][code] = group_name
            update.message.reply_text("Группа добавлена.")
        else:
            update.message.reply_text("Такая группа уже есть.")
        pupils.close()
        with open("info/pupils.json", "w", encoding="UTF-8") as pupils:
            json.dump(dct, pupils, ensure_ascii=False)
        pupils.close()


def sait(update, context):
    update.message.reply_text("https://yandex-master-final.herokuapp.com/")


def quit(update, context):
    inAny = False
    with open("info/pupils.json", "r", encoding='UTF-8') as f:
        dct = json.load(f)
    f.close()
    for i in dct["groups"].keys():
        if update.message.from_user.id in dct["groups"][i]:
            inAny = True
    if inAny:
        reply_keyboard = []
        for group in dct["groups"].keys():
            if update.message.from_user.id in dct["groups"][group]:
                reply_keyboard.append([group])
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        update.message.reply_text("Из какой группы вы хотите выйти?",
                                  reply_markup=markup)
    print("Дощел до quit ретурна")
    return 1


def strange(update, context):
    group = update.message.text
    with open("info/pupils.json", "r", encoding='UTF-8') as f:
        dct = json.load(f)
    f.close()
    if group not in dct["groups"].keys():
        update.message.reply_text("Группа не найдена.")
        return ConversationHandler.END
    if update.message.from_user.id in dct["groups"][group]:
        del dct["groups"][group][dct["groups"][group].index(update.message.from_user.id)]
        update.message.reply_text("Ученик удален.")
        with open("info/pupils.json", "w", encoding='UTF-8') as f:
            json.dump(dct, f, ensure_ascii=False)
    else:
        update.message.reply_text("Ученика нет в группе.")
    return ConversationHandler.END


def help(update, context):
    update.message.reply_text('''/set_timetable - Команда для установки расписания. 
    Синтаксис: /set_timetable <день недели> <предметы через пробел>
    -----------------------------------
    /timetable - Команда для просмотра расписания уроков.
    Синтаксис: /timetable <день недели>
    -----------------------------------
    /bell - Команда для просмотра расписания звонков
    Синтаксис: /bell
    -----------------------------------
    /add_teacher - Команда для добавления учителей (Только для учителей)
    Синтаксис: /add_teacher <пароль> <имя>
    -----------------------------------
    /send_messages - Команда для рассылки сообщений (Только для учителей)
    Синтаксис: /send_messages
    -----------------------------------
    /add_marks - Команда для добавления оценок в табель (Только для учителей)
    Синтаксис: /add_marks <имя ученика> <полугодие> <предмет> <оценки через пробел>
    -----------------------------------
    /add_pupil - Команда для добавления учеников
    Синтаксис: /add_pupil <имя>
    ----------------------------------------------------------------------
    /marks - Команда для просмтора табеля
    Синтаксис: /marks <имя> <полугодие цифрой 1 или 2>
    -----------------------------------
    /dont_send - Команда для отписки от рассылки
    Синтаксис: /send_messages
    -----------------------------------
    /sait - Сайт проекта
    Синтаксис: /sait
    -----------------------------------
    /add_group - Команда, добавляющая группу (Только для учителей)
    Синтаксис: /add_group <имя группы> <кодовое слово для добавления>
    -----------------------------------
    /quit - Покинуть группу
    Синтаксис: /quit <имя группы>
    -----------------------------------
    /help - Список команд
    Синтаксис: /help''')


text_handler = MessageHandler(Filters.text, check_codes)
conv_handler = ConversationHandler(
    entry_points=[CommandHandler('send_message', send_messages)],

    states={
        1: [MessageHandler(Filters.text, first_response)],
        2: [MessageHandler(Filters.text, second_response)]
    },

    fallbacks=[CommandHandler('stop', stop)]
)
quit_handler = ConversationHandler(
    entry_points=[CommandHandler('quit', quit)],

    states={
        1: [MessageHandler(Filters.text, strange)]
    },

    fallbacks=[CommandHandler('stop', stop)]
)
dp.add_handler(conv_handler)
dp.add_handler(quit_handler)
dp.add_handler(CommandHandler("set_timetable", set_timetable))
dp.add_handler(CommandHandler("timetable", timetable))
dp.add_handler(CommandHandler("bell", bell))
dp.add_handler(CommandHandler("add_teacher", add_teacher))
dp.add_handler(CommandHandler("add_marks", add_marks))
dp.add_handler(CommandHandler("marks", marks))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("sait", sait))
dp.add_handler(CommandHandler("mute", mute))
dp.add_handler(CommandHandler("add_group", add_group))
dp.add_handler(CommandHandler("quit", quit))
dp.add_handler(text_handler)

if __name__ == '__main__':
    main()

# для каждого пользователя добавляется понятие "добавиться в группу"
# можно удалиться из группы
# можно посмтотерть в какой группе ты состоишь
