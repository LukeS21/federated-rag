from src.unicode_map import scrub_unicode

def test_greek_and_subscript():
    input_text = "TNF-α levels in TiO₂ nanotubes"
    expected = "TNF-alpha levels in TiO2 nanotubes"
    assert scrub_unicode(input_text) == expected

def test_microgram():
    assert scrub_unicode("15 μg/mL") == "15 mcg/mL"