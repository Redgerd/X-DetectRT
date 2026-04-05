from sqlalchemy import Column, Integer, String, Boolean, Float, Text, DateTime, ForeignKey, Sequence
from models.base import Base
from datetime import datetime

class AudioAnalysis(Base):
    __tablename__ = 'audio_analysis'

    id = Column(Integer, Sequence('audio_analysis_seq'), primary_key=True, index=True)
    audio_file_id = Column(Integer, ForeignKey('audio_files.id'), nullable=False)
    verdict = Column(String, nullable=False)  # "FAKE" or "REAL"
    is_fake = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    fake_prob = Column(Float, nullable=False)
    real_prob = Column(Float, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    waveform_samples = Column(Text, nullable=True)  # JSON string of list[float]
    stft_data = Column(Text, nullable=True)  # JSON string of dict with matrix, times, freqs, db_min, db_max
    ig_scores = Column(Text, nullable=True)  # JSON string of list[float]
    shap_scores = Column(Text, nullable=True)  # JSON string of list[float]
    analysis_time = Column(DateTime, default=datetime.utcnow, nullable=False)