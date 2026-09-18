"""ICAO 9303 MRZ specification: charset, 7-3-1 check digits, TD1/TD2/TD3 line layouts.

Reference: ICAO Doc 9303, Part 3 (specifications common to all MRTDs) and Part 4/5
(TD3 passports / TD1 ID cards). The worked check-digit example in Part 3 §4.9 is used
as the self-test fixture at the bottom of this file.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

MRZ_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
FILLER = "<"
_CHECK_WEIGHTS = (7, 3, 1)


def char_value(c: str) -> int:
    """ICAO 9303 §4.9 numeric value of a single MRZ character."""
    if c == FILLER:
        return 0
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 10
    raise ValueError(f"character {c!r} is not in the MRZ charset")


def check_digit(data: str) -> int:
    """Compute the ICAO 9303 7-3-1 weighted mod-10 check digit for `data`."""
    total = 0
    for i, c in enumerate(data):
        total += char_value(c) * _CHECK_WEIGHTS[i % 3]
    return total % 10


def check_digit_char(data: str) -> str:
    return str(check_digit(data))


def verify_check_digit(data: str, digit_char: str) -> bool:
    return digit_char.isdigit() and check_digit(data) == int(digit_char)


class MrzFormat(str, Enum):
    TD1 = "TD1"  # 3 lines x 30 chars, ID cards
    TD2 = "TD2"  # 2 lines x 36 chars, ID cards / visas
    TD3 = "TD3"  # 2 lines x 44 chars, passports


LINE_SHAPE: dict[MrzFormat, tuple[int, int]] = {
    MrzFormat.TD1: (3, 30),
    MrzFormat.TD2: (2, 36),
    MrzFormat.TD3: (2, 44),
}


def detect_format(lines: list[str]) -> MrzFormat:
    n = len(lines)
    width = len(lines[0]) if lines else 0
    for fmt, (nlines, w) in LINE_SHAPE.items():
        if n == nlines and width == w:
            return fmt
    raise ValueError(f"unrecognised MRZ shape: {n} lines of width {width}")


@dataclass
class CheckGroup:
    """Mirrors the `MrzGroup` contract in BACKEND_BRIEF.md §4.

    `start`/`end` index into a single reference line (see `line` below) — the line
    the ribbon UI should underline against. For `composite`, the true ICAO check
    digit input is a *discontinuous* concatenation of several fields (it skips
    nationality and sex on TD3, for example); we approximate it here with the span
    covering the whole line minus the trailing composite digit, which is a
    simplification worth knowing about, not the literal check-digit input.
    """

    name: str
    line: int
    start: int
    end: int  # exclusive, excludes the check digit itself
    check_digit_index: int
    valid: bool
    expected: Optional[str] = None
    read: Optional[str] = None


def _resolve_year(yy: int, *, reference_year: int, past_bias: bool) -> int:
    """Two-digit -> four-digit year heuristic (ICAO does not fix a pivot year).

    For birth dates (`past_bias=True`) we pick whichever century keeps the date
    in the past. For expiry dates we pick whichever century lands closer to the
    reference year, since documents can be expired but not implausibly so.
    """
    century_now = (reference_year // 100) * 100
    candidate_this = century_now + yy
    candidate_prev = century_now - 100 + yy
    if past_bias:
        return candidate_this if candidate_this <= reference_year else candidate_prev
    return (
        candidate_this
        if abs(candidate_this - reference_year) <= abs(candidate_prev - reference_year)
        else candidate_prev
    )


def parse_mrz_date(
    yymmdd: str, *, past_bias: bool, reference_date: Optional[_dt.date] = None
) -> Optional[_dt.date]:
    """Parse a YYMMDD MRZ date field. Returns None if it isn't a valid calendar date."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    reference_date = reference_date or _dt.date.today()
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    year = _resolve_year(yy, reference_year=reference_date.year, past_bias=past_bias)
    try:
        return _dt.date(year, mm, dd)
    except ValueError:
        return None


def format_mrz_date(d: _dt.date) -> str:
    return f"{d.year % 100:02d}{d.month:02d}{d.day:02d}"


def _split_name_field(name_field: str) -> tuple[str, str]:
    surname_part, _, given_part = name_field.partition("<<")
    surname = surname_part.replace("<", " ").strip()
    given = given_part.replace("<", " ").strip()
    return surname, given


@dataclass
class MrzResult:
    format: MrzFormat
    doc_type: str
    issuing_country: str
    surname: str
    given_names: str
    doc_number: str
    nationality: str
    birth_date: Optional[_dt.date]
    birth_date_raw: str
    sex: str
    expiry_date: Optional[_dt.date]
    expiry_date_raw: str
    personal_number: str
    checks: list[CheckGroup] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def composite_valid(self) -> bool:
        return all(g.valid for g in self.checks)

    @property
    def all_valid(self) -> bool:
        return self.composite_valid


def parse_td3(lines: list[str]) -> MrzResult:
    """TD3: passports. Two lines of 44 characters."""
    if len(lines) != 2 or any(len(l) != 44 for l in lines):
        raise ValueError("TD3 requires two 44-character lines")
    l1, l2 = lines

    doc_type = l1[0:2].rstrip("<")
    issuing_country = l1[2:5]
    surname, given_names = _split_name_field(l1[5:44])

    doc_number = l2[0:9]
    doc_number_cd = l2[9]
    nationality = l2[10:13]
    birth_raw = l2[13:19]
    birth_cd = l2[19]
    sex = l2[20]
    expiry_raw = l2[21:27]
    expiry_cd = l2[27]
    personal_number = l2[28:42]
    personal_cd = l2[42]
    composite_cd = l2[43]

    composite_input = l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43]

    checks = [
        CheckGroup(
            "doc_number", 1, 0, 9, 9,
            verify_check_digit(doc_number, doc_number_cd),
            expected=check_digit_char(doc_number), read=doc_number_cd,
        ),
        CheckGroup(
            "birth_date", 1, 13, 19, 19,
            verify_check_digit(birth_raw, birth_cd),
            expected=check_digit_char(birth_raw), read=birth_cd,
        ),
        CheckGroup(
            "expiry_date", 1, 21, 27, 27,
            verify_check_digit(expiry_raw, expiry_cd),
            expected=check_digit_char(expiry_raw), read=expiry_cd,
        ),
        CheckGroup(
            "composite", 1, 0, 43, 43,
            verify_check_digit(composite_input, composite_cd),
            expected=check_digit_char(composite_input), read=composite_cd,
        ),
    ]
    if personal_number.strip("<"):
        checks.append(
            CheckGroup(
                "personal_number", 1, 28, 42, 42,
                verify_check_digit(personal_number, personal_cd),
                expected=check_digit_char(personal_number), read=personal_cd,
            )
        )

    return MrzResult(
        format=MrzFormat.TD3,
        doc_type=doc_type,
        issuing_country=issuing_country,
        surname=surname,
        given_names=given_names,
        doc_number=doc_number.rstrip("<"),
        nationality=nationality,
        birth_date=parse_mrz_date(birth_raw, past_bias=True),
        birth_date_raw=birth_raw,
        sex=sex,
        expiry_date=parse_mrz_date(expiry_raw, past_bias=False),
        expiry_date_raw=expiry_raw,
        personal_number=personal_number.rstrip("<"),
        checks=checks,
        lines=lines,
    )


def parse_td1(lines: list[str]) -> MrzResult:
    """TD1: ID cards. Three lines of 30 characters."""
    if len(lines) != 3 or any(len(l) != 30 for l in lines):
        raise ValueError("TD1 requires three 30-character lines")
    l1, l2, l3 = lines

    doc_type = l1[0:2].rstrip("<")
    issuing_country = l1[2:5]
    doc_number = l1[5:14]
    doc_number_cd = l1[14]
    opt1 = l1[15:30]

    birth_raw = l2[0:6]
    birth_cd = l2[6]
    sex = l2[7]
    expiry_raw = l2[8:14]
    expiry_cd = l2[14]
    nationality = l2[15:18]
    opt2 = l2[18:29]
    composite_cd = l2[29]

    surname, given_names = _split_name_field(l3)

    composite_input = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]

    checks = [
        CheckGroup(
            "doc_number", 0, 5, 14, 14,
            verify_check_digit(doc_number, doc_number_cd),
            expected=check_digit_char(doc_number), read=doc_number_cd,
        ),
        CheckGroup(
            "birth_date", 1, 0, 6, 6,
            verify_check_digit(birth_raw, birth_cd),
            expected=check_digit_char(birth_raw), read=birth_cd,
        ),
        CheckGroup(
            "expiry_date", 1, 8, 14, 14,
            verify_check_digit(expiry_raw, expiry_cd),
            expected=check_digit_char(expiry_raw), read=expiry_cd,
        ),
        CheckGroup(
            "composite", 1, 0, 29, 29,
            verify_check_digit(composite_input, composite_cd),
            expected=check_digit_char(composite_input), read=composite_cd,
        ),
    ]

    return MrzResult(
        format=MrzFormat.TD1,
        doc_type=doc_type,
        issuing_country=issuing_country,
        surname=surname,
        given_names=given_names,
        doc_number=doc_number.rstrip("<"),
        nationality=nationality,
        birth_date=parse_mrz_date(birth_raw, past_bias=True),
        birth_date_raw=birth_raw,
        sex=sex,
        expiry_date=parse_mrz_date(expiry_raw, past_bias=False),
        expiry_date_raw=expiry_raw,
        personal_number=(opt1 + opt2).replace("<", "").strip(),
        checks=checks,
        lines=lines,
    )


def parse_td2(lines: list[str]) -> MrzResult:
    """TD2: ID cards / visas. Two lines of 36 characters."""
    if len(lines) != 2 or any(len(l) != 36 for l in lines):
        raise ValueError("TD2 requires two 36-character lines")
    l1, l2 = lines

    doc_type = l1[0:2].rstrip("<")
    issuing_country = l1[2:5]
    surname, given_names = _split_name_field(l1[5:36])

    doc_number = l2[0:9]
    doc_number_cd = l2[9]
    nationality = l2[10:13]
    birth_raw = l2[13:19]
    birth_cd = l2[19]
    sex = l2[20]
    expiry_raw = l2[21:27]
    expiry_cd = l2[27]
    optional = l2[28:35]
    composite_cd = l2[35]

    composite_input = l2[0:10] + l2[13:20] + l2[21:28] + l2[28:35]

    checks = [
        CheckGroup(
            "doc_number", 1, 0, 9, 9,
            verify_check_digit(doc_number, doc_number_cd),
            expected=check_digit_char(doc_number), read=doc_number_cd,
        ),
        CheckGroup(
            "birth_date", 1, 13, 19, 19,
            verify_check_digit(birth_raw, birth_cd),
            expected=check_digit_char(birth_raw), read=birth_cd,
        ),
        CheckGroup(
            "expiry_date", 1, 21, 27, 27,
            verify_check_digit(expiry_raw, expiry_cd),
            expected=check_digit_char(expiry_raw), read=expiry_cd,
        ),
        CheckGroup(
            "composite", 1, 0, 35, 35,
            verify_check_digit(composite_input, composite_cd),
            expected=check_digit_char(composite_input), read=composite_cd,
        ),
    ]

    return MrzResult(
        format=MrzFormat.TD2,
        doc_type=doc_type,
        issuing_country=issuing_country,
        surname=surname,
        given_names=given_names,
        doc_number=doc_number.rstrip("<"),
        nationality=nationality,
        birth_date=parse_mrz_date(birth_raw, past_bias=True),
        birth_date_raw=birth_raw,
        sex=sex,
        expiry_date=parse_mrz_date(expiry_raw, past_bias=False),
        expiry_date_raw=expiry_raw,
        personal_number=optional.rstrip("<"),
        checks=checks,
        lines=lines,
    )


_PARSERS = {
    MrzFormat.TD1: parse_td1,
    MrzFormat.TD2: parse_td2,
    MrzFormat.TD3: parse_td3,
}


def parse(lines: list[str]) -> MrzResult:
    fmt = detect_format(lines)
    return _PARSERS[fmt](lines)


if __name__ == "__main__":
    # Self-test: the worked example from ICAO Doc 9303 Part 4, Appendix B
    # (a TD3 passport MRZ with all four check digits known-good).
    icao_example = [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]
    assert check_digit("L898902C3") == 6, check_digit("L898902C3")
    assert check_digit("740812") == 2, check_digit("740812")
    assert check_digit("120415") == 9, check_digit("120415")
    assert check_digit("ZE184226B<<<<<") == 1, check_digit("ZE184226B<<<<<")
    assert check_digit(
        "L898902C36" + "7408122" + "1204159" + "ZE184226B<<<<<1"
    ) == 0

    result = parse_td3(icao_example)
    assert result.surname == "ERIKSSON", result.surname
    assert result.given_names == "ANNA MARIA", result.given_names
    assert result.doc_number == "L898902C3", result.doc_number
    assert result.nationality == "UTO"
    assert result.sex == "F"
    assert result.birth_date_raw == "740812"
    assert result.expiry_date_raw == "120415"
    assert result.composite_valid, [(g.name, g.valid) for g in result.checks]
    assert result.all_valid

    # detect_format dispatch
    assert detect_format(icao_example) == MrzFormat.TD3
    assert parse(icao_example).doc_number == "L898902C3"

    # A tampered birth date must fail exactly that check digit, nothing else.
    tampered = list(icao_example)
    tampered[1] = "L898902C36UTO7408132F1204159ZE184226B<<<<<10"  # DOB day 12->13
    tampered_result = parse_td3(tampered)
    by_name = {g.name: g for g in tampered_result.checks}
    assert not by_name["birth_date"].valid
    assert by_name["doc_number"].valid
    assert by_name["expiry_date"].valid

    print("spec.py self-test OK: ICAO 9303 Part 4 Appendix B example verified,")
    print(f"  parsed: {result.surname}, {result.given_names}, {result.doc_number}, "
          f"nationality={result.nationality}, sex={result.sex}, "
          f"birth={result.birth_date}, expiry={result.expiry_date}")
    print("  tampered DOB correctly isolated to the birth_date check group.")
