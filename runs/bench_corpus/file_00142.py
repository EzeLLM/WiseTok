from flask.cli import FlaskGroup
from project import create_app, db
from project.delijn.stop_data import get_stop_data
from project.delijn.vehicle_data import get_vehicle_data
from project.models.Vehicle import Vehicle
from project.models.Stop import Stop

app = create_app()
cli = FlaskGroup(create_app=create_app)


@cli.command('reset')
def recreate_db():
    """Resets the database: deletes all entries and recreates the models"""
    db.drop_all()
    db.create_all()
    db.session.commit()
    print("Database recreated")


def _add_vehicle_samples():
    data = get_vehicle_data()
    for v in data:
        vehicle = Vehicle(id=v.get('id'), number=v.get('number'), description=v.get('description'),
                          vehicle_type=v.get('type'), user_id=v.get('created_by'), name=v.get('name'))
        db.session.add(vehicle)


@cli.command("add_vehicles")
def add_vehicle_samples():
    """Adds vehicle samples to the database"""
    _add_vehicle_samples()
    db.session.commit()
    print("Added Vehicle samples")


def _add_stop_data():
    data = get_stop_data()
    for s in data:
        stop = Stop(name=s.get('name'), region=s.get('region'), stop_number=s.get('number'), village=s.get('village'))
        db.session.add(stop)


@cli.command('add_stops')
def add_stop_data():
    """Adds stop samples to the database"""
    _add_stop_data()
    db.session.commit()
    print("Added Stop samples")


@cli.command('fill_db')
def fill_db():
    """Fills the database with the stops of de lijn and dummy vehicle data"""
    _add_vehicle_samples()
    _add_stop_data()
    db.session.commit()
    print("Filled database")


if __name__ == '__main__':
    cli()
