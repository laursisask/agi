package xyz.laur.duels;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SkinChangerTest {
    @Test
    public void testSkinCount() {
        assertEquals(300, SkinChanger.SKINS.length);
    }
}
