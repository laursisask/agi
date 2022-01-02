package xyz.laur.duels;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class EmblemRendererTest {
    @Test
    public void testEmblemCount() {
        assertEquals(37, EmblemRenderer.EMBLEMS.length);
    }
}
