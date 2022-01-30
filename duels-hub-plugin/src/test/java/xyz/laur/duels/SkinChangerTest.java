package xyz.laur.duels;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SkinChangerTest {
    @Test
    public void testSkinCount() {
        assertEquals(10000, SkinChanger.SKINS.length);
    }
}
