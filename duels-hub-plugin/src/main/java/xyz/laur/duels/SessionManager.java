package xyz.laur.duels;

import org.bukkit.entity.Player;

import java.util.HashMap;
import java.util.Map;

public class SessionManager {
    private final Map<String, GameSession> games = new HashMap<>();

    public GameSession getByName(String name) {
        return games.get(name);
    }

    public void addSession(String name, GameSession session) {
        if (games.containsKey(name)) {
            throw new IllegalArgumentException("Session with such name already exists");
        }

        games.put(name, session);
    }

    public void removeSession(GameSession session) {
        games.values().removeIf(s -> s.equals(session));
    }

    public GameSession getPlayerSession(Player player) {
        for (GameSession session : games.values()) {
            if (session.hasPlayer(player)) {
                return session;
            }
        }

        return null;
    }

    public Map<String, GameSession> getAll() {
        return games;
    }
}
