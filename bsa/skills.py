"""What the agent does for each of the 77 intents.

Every intent maps to exactly one skill. Skills that read account data call read tools. Skills that change
something (freeze, activate, cancel, dispute) always go select -> confirm -> one-time passcode -> act.
Anything else is answered from the help-center article for that intent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str
    writes: bool = False          # changes account state: needs confirmation + OTP
    handoff_queue: str | None = None
    priority: str = "normal"


SKILLS = {
    "answer": Skill("answer"),
    "track_card": Skill("track_card"),
    "freeze_card": Skill("freeze_card", writes=True),
    "activate_card": Skill("activate_card", writes=True),
    "transfer_status": Skill("transfer_status"),
    "cancel_transfer": Skill("cancel_transfer", writes=True),
    "declined_payment": Skill("declined_payment"),
    "pending_payment": Skill("pending_payment"),
    "dispute_unrecognised": Skill("dispute_unrecognised", writes=True, handoff_queue="fraud", priority="high"),
    "dispute_duplicate": Skill("dispute_duplicate", writes=True),
    "top_up_status": Skill("top_up_status"),
    "exchange_rate": Skill("exchange_rate"),
    "handoff_closure": Skill("handoff_closure", handoff_queue="account-closure"),
    "handoff_security": Skill("handoff_security", handoff_queue="security", priority="high"),
}

INTENT_SKILL = {
    "card_arrival": "track_card", "card_delivery_estimate": "track_card",
    "lost_or_stolen_card": "freeze_card", "compromised_card": "freeze_card", "card_swallowed": "freeze_card",
    "activate_my_card": "activate_card",
    "pending_transfer": "transfer_status", "transfer_not_received_by_recipient": "transfer_status",
    "balance_not_updated_after_bank_transfer": "transfer_status", "failed_transfer": "transfer_status",
    "declined_transfer": "transfer_status",
    "cancel_transfer": "cancel_transfer",
    "declined_card_payment": "declined_payment", "declined_cash_withdrawal": "declined_payment",
    "pending_card_payment": "pending_payment", "pending_cash_withdrawal": "pending_payment",
    "card_payment_not_recognised": "dispute_unrecognised", "cash_withdrawal_not_recognised": "dispute_unrecognised",
    "direct_debit_payment_not_recognised": "dispute_unrecognised",
    "transaction_charged_twice": "dispute_duplicate",
    "pending_top_up": "top_up_status", "top_up_failed": "top_up_status", "top_up_reverted": "top_up_status",
    "verify_top_up": "top_up_status",
    "exchange_rate": "exchange_rate",
    "terminate_account": "handoff_closure",
    "lost_or_stolen_phone": "handoff_security",
}

# How an intent is offered to the customer when the agent asks "Did you mean...?"
LABELS = {
    "Refund_not_showing_up": "A refund that hasn't arrived", "activate_my_card": "Activating my card",
    "age_limit": "Age requirements", "apple_pay_or_google_pay": "Apple Pay or Google Pay", "atm_support": "Using ATMs",
    "automatic_top_up": "Automatic top-ups", "balance_not_updated_after_bank_transfer": "Balance not updated after a transfer",
    "balance_not_updated_after_cheque_or_cash_deposit": "Balance not updated after a deposit",
    "beneficiary_not_allowed": "Can't add a recipient", "cancel_transfer": "Cancelling a transfer",
    "card_about_to_expire": "My card is about to expire", "card_acceptance": "Where my card is accepted",
    "card_arrival": "Where my new card is", "card_delivery_estimate": "How long card delivery takes",
    "card_linking": "Linking a card", "card_not_working": "My card isn't working",
    "card_payment_fee_charged": "A fee on a card payment", "card_payment_not_recognised": "A card payment I don't recognise",
    "card_payment_wrong_exchange_rate": "Wrong exchange rate on a payment", "card_swallowed": "An ATM kept my card",
    "cash_withdrawal_charge": "ATM withdrawal fees", "cash_withdrawal_not_recognised": "A cash withdrawal I don't recognise",
    "change_pin": "Changing my PIN", "compromised_card": "My card may be compromised",
    "contactless_not_working": "Contactless isn't working", "country_support": "Supported countries",
    "declined_card_payment": "A declined card payment", "declined_cash_withdrawal": "A declined ATM withdrawal",
    "declined_transfer": "A declined transfer", "direct_debit_payment_not_recognised": "A direct debit I don't recognise",
    "disposable_card_limits": "Disposable card limits", "edit_personal_details": "Changing my details",
    "exchange_charge": "Currency exchange fees", "exchange_rate": "Exchange rates", "exchange_via_app": "Exchanging currency in the app",
    "extra_charge_on_statement": "An extra charge on my statement", "failed_transfer": "A failed transfer",
    "fiat_currency_support": "Supported currencies", "get_disposable_virtual_card": "Getting a disposable card",
    "get_physical_card": "Getting a physical card", "getting_spare_card": "Getting a spare card",
    "getting_virtual_card": "Getting a virtual card", "lost_or_stolen_card": "My card is lost or stolen",
    "lost_or_stolen_phone": "My phone is lost or stolen", "order_physical_card": "Ordering a physical card",
    "passcode_forgotten": "I forgot my passcode", "pending_card_payment": "A pending card payment",
    "pending_cash_withdrawal": "A pending ATM withdrawal", "pending_top_up": "A pending top-up",
    "pending_transfer": "A pending transfer", "pin_blocked": "My PIN is blocked", "receiving_money": "Receiving money",
    "request_refund": "Getting a refund", "reverted_card_payment?": "A reverted card payment",
    "supported_cards_and_currencies": "Supported cards and currencies", "terminate_account": "Closing my account",
    "top_up_by_bank_transfer_charge": "Bank transfer top-up fees", "top_up_by_card_charge": "Card top-up fees",
    "top_up_by_cash_or_cheque": "Depositing cash or cheques", "top_up_failed": "A failed top-up",
    "top_up_limits": "Top-up limits", "top_up_reverted": "A reverted top-up", "topping_up_by_card": "Topping up by card",
    "transaction_charged_twice": "I was charged twice", "transfer_fee_charged": "A transfer fee",
    "transfer_into_account": "Transferring money in", "transfer_not_received_by_recipient": "A transfer the recipient didn't get",
    "transfer_timing": "How long transfers take", "unable_to_verify_identity": "I can't verify my identity",
    "verify_my_identity": "Verifying my identity", "verify_source_of_funds": "Proving where my money came from",
    "verify_top_up": "Verifying a top-up", "virtual_card_not_working": "My virtual card isn't working",
    "visa_or_mastercard": "Visa or Mastercard", "why_verify_identity": "Why I need to verify my identity",
    "wrong_amount_of_cash_received": "An ATM gave the wrong amount", "wrong_exchange_rate_for_cash_withdrawal": "Wrong rate on an ATM withdrawal",
}


def skill_for(intent: str) -> Skill:
    return SKILLS[INTENT_SKILL.get(intent, "answer")]


def label(intent: str) -> str:
    return LABELS.get(intent, intent.replace("_", " ").capitalize())
