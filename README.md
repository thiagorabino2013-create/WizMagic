# Wizarding Magic — Minecraft 1.21.11 (Fabric)

Varinhas e feitiços com efeitos 100 % originais (partículas, texturas, sons e animações próprios; nenhuma partícula,
relâmpago ou explosão vanilla).

## Compilar o `.jar`

Requisitos: **JDK 21** e acesso à internet (o Gradle descarrega o Minecraft, o Fabric Loom e o Fabric API).

```
gradle wrapper --gradle-version 9.2.1     # só uma vez, se tiveres o Gradle instalado
./gradlew build                           # Windows: gradlew.bat build
```

O mod fica em `build/libs/wizmagic-1.0.0.jar` (ignora o `-sources`). Põe-no na pasta `mods` com o
**Fabric Loader ≥ 0.18** e o **Fabric API 0.141.1+1.21.11** (ou mais recente para 1.21.11).
Em IntelliJ: abre a pasta como projeto Gradle e corre a tarefa `build` (ou `runClient` para testar).

Se o Gradle não encontrar alguma versão, vê os valores atuais em https://fabricmc.net/develop e ajusta o `gradle.properties`.

## Como jogar

- **Varinhas** (6): separador Ferramentas/Combate do criativo, ou receita `  C / S / S` (core + 2 paus). Cada uma tem madeira, núcleo e cor próprios.
- **Clique direito** lança o feitiço selecionado (carregamento curto, depois a esfera sai da varinha na direção da mira).
- **Teclas**: `Z` Água · `X` Relâmpago · `C` Inferno · `V` Expulso · `G` abre o ecrã de seleção (tudo reconfigurável em Controlos).
- HUD no canto inferior direito com feitiço selecionado, cooldowns e energia mágica.

## Configuração

`config/wizmagic.json` (criado no primeiro arranque): dano, alcance, velocidade, custo de mana, cooldown, raio,
quebra de blocos (`maxBlockHardness`, `blockBlacklist`, `blockWhitelist`), `particleDensity` (0.25–3) e mais.

## Regenerar texturas, modelos e sons

`python3 tools/generate_assets.py` (precisa de numpy, scipy, Pillow e ffmpeg com libvorbis).
