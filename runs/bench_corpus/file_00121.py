def passenger_walk(trains, passenger_name, steps):
    for train in trains:
        for car_index, car in enumerate(train['cars']):
            if passenger_name in car['people']:
                if 0 <= steps + car_index <= len(train['cars']):
                    car['people'].remove(passenger_name)
                    train['cars'][car_index + steps]['people'].append(passenger_name)
                    return 1
                else:
                    return -1
    return -1


def cars_switch(event, trains):
    first_train = None
    second_train = None
    for train in trains:

        if train['name'] == event['train_from']:
            first_train = train
        if train['name'] == event['train_to']:
            second_train = train

    if second_train and first_train and 1 <= event['cars'] <= len(first_train['cars']):
        coupling_cars = first_train['cars'][-event['cars']:]
        first_train['cars'] = first_train['cars'][:-event['cars']]
        second_train['cars'].extend(coupling_cars)
        return 1
    else:
        return -1


def process(data, events, car):
    for event in events:
        if event['type'] == 'walk':

            if passenger_walk(data, event['passenger'], event['distance']) == -1:
                return -1

        elif event['type'] == 'switch':

            if cars_switch(event, data) == -1:
                return -1
        else:
            return -1

    for train in data:
        print(train['name'])
        for car1 in train['cars']:
            print('\t{}'.format(car1['name']))
            for man in car1['people']:
                print('\t\t{}'.format(man))

    for dat in data:
        for carriage in dat['cars']:
            if car == carriage['name']:
                return len(carriage['people'])

    return -1
