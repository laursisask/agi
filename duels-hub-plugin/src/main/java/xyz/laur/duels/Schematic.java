package xyz.laur.duels;

import net.minecraft.server.v1_8_R3.NBTCompressedStreamTools;
import net.minecraft.server.v1_8_R3.NBTTagCompound;
import org.bukkit.Material;

import java.io.IOException;
import java.io.InputStream;

public class Schematic {
    private final short width;
    private final short height;
    private final short length;
    private final Material[][][] blocks;
    private final byte[][][] blockDatas;

    public Schematic(short width, short height, short length, Material[][][] blocks, byte[][][] blockDatas) {
        this.width = width;
        this.height = height;
        this.length = length;
        this.blocks = blocks;
        this.blockDatas = blockDatas;
    }

    public short getWidth() {
        return width;
    }

    public short getHeight() {
        return height;
    }

    public short getLength() {
        return length;
    }

    public Material[][][] getBlocks() {
        return blocks;
    }

    public byte[][][] getBlockDatas() {
        return blockDatas;
    }

    public static Schematic parse(InputStream stream) throws IOException {
        NBTTagCompound root = NBTCompressedStreamTools.a(stream);
        short width = root.getShort("Width");
        short height = root.getShort("Height");
        short length = root.getShort("Length");

        byte[] blockIdsRaw = root.getByteArray("Blocks");
        byte[] datasFlat = root.getByteArray("Data");

        int[] blockIdsFlat = new int[blockIdsRaw.length];
        for (int i = 0; i < blockIdsRaw.length; i++) {
            blockIdsFlat[i] = Byte.toUnsignedInt(blockIdsRaw[i]);
        }

        Material[][][] blocks = new Material[width][height][length];
        byte[][][] blockDatas = new byte[width][height][length];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < length; z++) {
                    int flatIndex = (y * length + z) * width + x;
                    //noinspection deprecation
                    blocks[x][y][z] = Material.getMaterial(blockIdsFlat[flatIndex]);

                    blockDatas[x][y][z] = datasFlat[flatIndex];
                }
            }
        }

        return new Schematic(width, height, length, blocks, blockDatas);
    }
}
