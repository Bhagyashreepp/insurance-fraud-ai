"""SQLAlchemy models for the synthetic insurance dataset."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database tables."""


class TimestampMixin:
    """Creation and update timestamps stored on each persisted entity."""

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class Agent(TimestampMixin, Base):
    __tablename__ = "agents"

    agent_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(160), nullable=False)
    region: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    hire_date: Mapped[date] = mapped_column(Date, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    claims: Mapped[list[Claim]] = relationship(back_populates="agent")


class Customer(TimestampMixin, Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(160), nullable=False)
    state: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    customer_since: Mapped[date] = mapped_column(Date, nullable=False)

    policies: Mapped[list[Policy]] = relationship(back_populates="customer")
    claims: Mapped[list[Claim]] = relationship(back_populates="customer")


class Policy(TimestampMixin, Base):
    __tablename__ = "policies"
    __table_args__ = (Index("ix_policies_customer_status", "customer_id", "policy_status"),)

    policy_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False, index=True)
    policy_type: Mapped[str] = mapped_column(String(32), nullable=False)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    expiration_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    policy_status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    coverage_limit: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="policies")
    claims: Mapped[list[Claim]] = relationship(back_populates="policy")


class Contractor(TimestampMixin, Base):
    __tablename__ = "contractors"

    contractor_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    contractor_name: Mapped[str] = mapped_column(String(160), nullable=False)
    specialty: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    licensed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    claims: Mapped[list[Claim]] = relationship(back_populates="contractor")


class Claim(TimestampMixin, Base):
    __tablename__ = "claims"
    __table_args__ = (
        Index("ix_claims_agent_submitted", "agent_id", "submitted_date"),
        Index("ix_claims_customer_submitted", "customer_id", "submitted_date"),
        Index("ix_claims_policy_incident", "policy_id", "incident_date"),
    )

    claim_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    policy_id: Mapped[str] = mapped_column(ForeignKey("policies.policy_id"), nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.agent_id"), nullable=False, index=True)
    contractor_id: Mapped[str] = mapped_column(ForeignKey("contractors.contractor_id"), nullable=False, index=True)
    loss_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    incident_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    submitted_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    claim_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    fraud_label: Mapped[int] = mapped_column(Integer, nullable=False)
    injected_patterns: Mapped[str] = mapped_column(Text, nullable=False, default="")
    duplicate_of_claim_id: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    policy: Mapped[Policy] = relationship(back_populates="claims")
    customer: Mapped[Customer] = relationship(back_populates="claims")
    agent: Mapped[Agent] = relationship(back_populates="claims")
    contractor: Mapped[Contractor] = relationship(back_populates="claims")
    documents: Mapped[list[Document]] = relationship(back_populates="claim")


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_claim_document_date", "claim_id", "document_date"),)

    document_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id"), nullable=False, index=True)
    document_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    document_date: Mapped[date] = mapped_column(Date, nullable=False)
    uploaded_date: Mapped[date] = mapped_column(Date, nullable=False)
    document_reference: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)

    claim: Mapped[Claim] = relationship(back_populates="documents")
