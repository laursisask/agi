package xyz.laur.duels;

import org.junit.Test;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class SchematicTest {
    @Test
    public void testParse() throws Exception {
        InputStream stream = SchematicTest.class.getClassLoader().getResourceAsStream("emblems/Fire.schematic");

        Schematic schematic = Schematic.parse(stream);

        assertEquals(1, schematic.getWidth());
        assertEquals(14, schematic.getHeight());
        assertEquals(12, schematic.getLength());

        List<String> expectedMaterials = Arrays.asList(
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSSSSSSSS",
                "SSSSSGSGSSSS",
                "SSSSGGGGSSSS",
                "SSSSSSSSSSSS"
        );

        assertEquals(expectedMaterials, get2dMaterials(schematic));

        List<String> expectedDatas = Arrays.asList(
                "GGGGGGGGGGGG",
                "GBBBRRBBBBBG",
                "GBBBBRRRBBBG",
                "GBRBBBRORBBG",
                "GBBRBROORBBG",
                "GBRBBRORRBBG",
                "GBRRROYRBBRG",
                "GRRORROORRRG",
                "GROYOOYYORBG",
                "GROOYYOYOYRG",
                "GROYOYOYYORG",
                "GBRYY-Y-YRBG",
                "GBBR----RBBG",
                "GGGGGGGGGGGG"
        );

        assertEquals(expectedDatas, get2dDatas(schematic));
    }

    private List<String> get2dMaterials(Schematic schematic) {
        List<String> materials = new ArrayList<>();

        for (int y = schematic.getHeight() - 1; y >= 0; y--) {
            StringBuilder line = new StringBuilder();

            for (int z = 0; z < schematic.getLength(); z++) {
                line.append(schematic.getBlocks()[0][y][z].toString().charAt(0));
            }

            materials.add(line.toString());
        }

        return materials;
    }

    private List<String> get2dDatas(Schematic schematic) {
        List<String> materials = new ArrayList<>();

        for (int y = schematic.getHeight() - 1; y >= 0; y--) {
            StringBuilder line = new StringBuilder();

            for (int z = 0; z < schematic.getLength(); z++) {
                byte data = schematic.getBlockDatas()[0][y][z];
                line.append(getDataCode(data));
            }

            materials.add(line.toString());
        }

        return materials;
    }

    private char getDataCode(byte data) {
        switch (data) {
            // Default
            case 0:
                return '-';
            // Orange
            case 1:
                return 'O';
            // Yellow
            case 4:
                return 'Y';
            // Grey
            case 7:
                return 'G';
            // Red
            case 14:
                return 'R';
            // Black
            case 15:
                return 'B';

            default:
                throw new UnsupportedOperationException("Unknown data " + data);
        }
    }
}
