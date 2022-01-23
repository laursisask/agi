package xyz.laur.duels;

import org.bukkit.Bukkit;
import org.bukkit.Server;
import org.bukkit.craftbukkit.v1_8_R3.inventory.CraftItemFactory;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.enchantments.EnchantmentTarget;
import org.bukkit.inventory.ItemStack;

import java.util.logging.Logger;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class TestHelper {
    public static void mockServer() {
        if (Bukkit.getServer() != null) return;

        Server server = mock(Server.class);
        when(server.getLogger()).thenReturn(Logger.getGlobal());
        when(server.getName()).thenReturn("Mock Server");
        when(server.getVersion()).thenReturn("1.8.8");
        when(server.getBukkitVersion()).thenReturn("1.8.8");
        Bukkit.setServer(server);
        when(server.getItemFactory()).thenReturn(CraftItemFactory.instance());

        setUpEnchantments();
    }

    private static void setUpEnchantments() {
        Enchantment protection = new Enchantment(0) {
            @Override
            public String getName() {
                return "PROTECTION_ENVIRONMENTAL";
            }

            @Override
            public int getMaxLevel() {
                return 4;
            }

            @Override
            public int getStartLevel() {
                return 1;
            }

            @Override
            public EnchantmentTarget getItemTarget() {
                return EnchantmentTarget.ARMOR;
            }

            @Override
            public boolean conflictsWith(Enchantment other) {
                return false;
            }

            @Override
            public boolean canEnchantItem(ItemStack item) {
                return true;
            }
        };

        Enchantment.registerEnchantment(protection);
    }

}
