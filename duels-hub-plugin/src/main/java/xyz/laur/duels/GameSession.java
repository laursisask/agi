package xyz.laur.duels;

import org.bukkit.entity.Player;

import java.util.List;

public interface GameSession {
    boolean hasPlayer(Player player);
    List<Player> getPlayers();
    GameState getState();
    String getMapName();
}
