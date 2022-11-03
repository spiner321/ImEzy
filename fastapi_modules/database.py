from sqlalchemy import create_engine, Boolean, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


SQLALCHEMY_DATABASE_URI = 'sqlite:////data/kimgh/ImEzy/stable-diffusion-flask/DB/imezy_api.db'

engine = create_engine(
    SQLALCHEMY_DATABASE_URI, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Input_Info_DB(Base):
    __tablename__ = 'input_info'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    uuid = Column(String, unique=True)
    prompt = Column(String)
    negative_prompt = Column(String)
    seed = Column(Integer)
    num_cnt_seed = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    sampler = Column(String)
    cfg_scale = Column(Float)
    steps = Column(Integer)
    batch_size = Column(Integer)
    denoising_strength = Column(Float)
    datetime = Column(DateTime)
    type = Column(String)

    
def insert_db(db, username, uuid, info, datetime, type):
    db.username = username
    db.uuid = uuid
    db.prompt = info['prompt']
    db.negative_prompt = info['negative_prompt']
    db.seed = info['seed']
    db.num_cnt_seed = len(info['all_seeds'])
    db.width = info['width']
    db.height = info['height']
    db.sampler = info['sampler']
    db.cfg_scale = info['cfg_scale']
    db.steps = info['steps']
    db.batch_size = info['batch_size']
    db.denoising_strength = info['denoising_strength']
    db.datetime = datetime
    db.type = type

    return db