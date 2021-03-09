using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

using BayatGames.SaveGameFree;
using BayatGames.SaveGameFree.Serializers;

using RedRunner.Characters;
using RedRunner.Collectables;
using RedRunner.TerrainGeneration;
using UnityEngine.Analytics;
using System.IO;
using System.Text;

namespace RedRunner
{
    public sealed class GameManager : MonoBehaviour
    {
        public delegate void AudioEnabledHandler(bool active);

        public delegate void ScoreHandler(float newScore, float highScore, float lastScore);

        public delegate void ResetHandler();

        public delegate void TimerHandler(float newTime);

        public static event ResetHandler OnReset;
        public static event ScoreHandler OnScoreChanged;
        public static event TimerHandler OnTimerChanged;
        public static event AudioEnabledHandler OnAudioEnabled;

        private static GameManager m_Singleton;

        public static GameManager Singleton
        {
            get
            {
                return m_Singleton;
            }
        }

        [SerializeField]
        private Character m_MainCharacter;
        [SerializeField]
        [TextArea(3, 30)]
        private string m_ShareText;
        [SerializeField]
        private string m_ShareUrl;
        private float m_StartScoreX = 0f;
        private float m_HighScore = 0f;
        private float m_LastScore = 0f;
        private float m_Score = 0f;
        private float m_Timer = 300f;

        private bool m_GameStarted = false;
        private bool m_GameRunning = false;
        private bool m_AudioEnabled = true;

        #region Analitics

        private Block current_block;
        private float current_block_position;
        private int game_id;

        // Coins
        private int coins_game;
        private int coins_section;
        private int coins_easy_game;
        private int coins_easy_section;
        private int coins_hard_game;
        private int coins_hard_section;
        private int chest_normal_game;
        private int chest_normal_section;
        private int chest_hard_game;
        private int chest_hard_section;
        // Time
        private float total_time_game;
        private float total_time_section;
        private float total_time_gained_game;
        private float total_time_gained_section;
        // Lives
        private int lives_lost_game;
        private int lives_lost_section;
        private int lives_lost_enemy_static_game;
        private int lives_lost_enemy_static_section;
        private int lives_lost_enemy_moving_game;
        private int lives_lost_enemy_moving_section;
        private int lives_lost_enemy_water_game;
        private int lives_lost_enemy_water_section;
        // Score
        private float best_score_game;
        private float score_game;
        private float score_section;
        // Movement
        private int player_backtracked_game;
        private bool player_backtracked_section;
        private int jumps_game;
        private int jumps_section;
        private float average_velocity_game;
        private int average_velocity_game_i;
        private float average_velocity_section;
        private int average_velocity_section_i;
        [SerializeField]
        private string csv_section_path = @"./data_section.csv";
        [SerializeField]
        private string csv_game_path = @"./data_game.csv";
        private StreamWriter csv_section;
        private StreamWriter csv_game;
        private bool live_lost = false;

        #endregion


        /// <summary>
        /// This is my developed callbacks compoents, because callbacks are so dangerous to use we need something that automate the sub/unsub to functions
        /// with this in-house developed callbacks feature, we garantee that the callback will be removed when we don't need it.
        /// </summary>
        public Property<int> m_Coin = new Property<int>(0);

        public float max_game_time = 300f;
        public float coin_add_game_time = 1f;


        #region Getters
        public bool gameStarted
        {
            get
            {
                return m_GameStarted;
            }
        }

        public bool gameRunning
        {
            get
            {
                return m_GameRunning;
            }
        }

        public bool audioEnabled
        {
            get
            {
                return m_AudioEnabled;
            }
        }
        public int coinsGame
        {
            get
            {
                return coins_game;
            }
        }
        public float currentScore
        {
            get
            {
                return m_Score;
            }
        }
        public float bestScore
        {
            get
            {
                return m_HighScore;
            }
        }
        public float totalGameTime
        {
            get
            {
                return total_time_game;
            }
        }
        public float currentGameTime
        {
            get
            {
                return m_Timer;
            }
        }
        #endregion

        public float Timer
        {
            get
            {
                return m_Timer;
            }
            set
            {
                m_Timer = value;
                if (m_Timer <= 0)
                {
                    m_Timer = 0;
                    StartCoroutine(DeathCrt(0.1f));
                }
                if (OnTimerChanged != null)
                {
                    OnTimerChanged(m_Timer);
                }
            }
        }

        void Awake()
        {
            System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
            customCulture.NumberFormat.NumberDecimalSeparator = ".";
            System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;


            if (m_Singleton != null)
            {
                Destroy(gameObject);
                return;
            }
            SaveGame.Serializer = new SaveGameBinarySerializer();
            m_Singleton = this;
            m_Score = 0f;

            if (SaveGame.Exists("coin"))
            {
                m_Coin.Value = SaveGame.Load<int>("coin");
            }
            else
            {
                m_Coin.Value = 0;
            }
            if (SaveGame.Exists("id"))
            {
                game_id = SaveGame.Load<int>("id") + 1;
            }
            else
            {
                game_id = 0;
            }
            if (SaveGame.Exists("audioEnabled"))
            {
                SetAudioEnabled(SaveGame.Load<bool>("audioEnabled"));
            }
            else
            {
                SetAudioEnabled(true);
            }
            if (SaveGame.Exists("lastScore"))
            {
                m_LastScore = SaveGame.Load<float>("lastScore");
            }
            else
            {
                m_LastScore = 0f;
            }
            if (SaveGame.Exists("highScore"))
            {
                m_HighScore = SaveGame.Load<float>("highScore");
            }
            else
            {
                m_HighScore = 0f;
            }
            if (!File.Exists(csv_section_path))
            {
                this.csv_section = File.AppendText(csv_section_path);
                this.csv_section.WriteLine("unique_identifier;session_id;section;total_coins;captured_coins;easy_captured_coins;hard_captured_coins;chest_normals;chest_hard;total_time;total_time_gained;lives_lost;lives_lost_water;lives_lost_static;lives_lost_moving;best_score;end_score;player_backtracked;jumps;average_velocity");
            }
            else this.csv_section = File.AppendText(csv_section_path);
            if (!File.Exists(csv_game_path))
            {
                this.csv_game = File.AppendText(csv_game_path);
                this.csv_game.WriteLine("unique_identifier;session_id;total_coins;captured_coins;easy_captured_coins;hard_captured_coins;chest_normals;chest_hard;total_time;total_time_gained;lives_lost;lives_lost_water;lives_lost_static;lives_lost_moving;best_score;end_score;player_backtracked;jumps;average_velocity");
            }
            else this.csv_game = File.AppendText(csv_game_path);
        }

        void UpdateDeathEvent(bool isDead)
        {
            if (isDead)
            {
                StartCoroutine(DeathCrt(1.5f));
            }
            else
            {
                StopCoroutine("DeathCrt");
            }
        }
        public void CoinCollection(bool hard)
        {
            if (m_GameRunning)
            {
                coins_game++;
                coins_section++;
                if (hard)
                {
                    coins_hard_game++;
                    coins_hard_section++;
                }
                else
                {
                    coins_easy_game++;
                    coins_easy_section++;
                }
                m_Timer += coin_add_game_time;
                total_time_gained_game += coin_add_game_time;
                total_time_gained_section += coin_add_game_time;
                if (m_Timer > max_game_time)
                {
                    m_Timer = max_game_time;
                }
                if (OnTimerChanged != null)
                {
                    OnTimerChanged(m_Timer);
                }
            }
        }

        public void ChestCollection(bool hard)
        {
            if (m_GameRunning)
            {
                if (hard)
                {
                    chest_hard_game++;
                    chest_hard_section++;
                }
                else
                {
                    chest_normal_game++;
                    chest_normal_section++;
                }
            }
        }

        public void LiveLost(int reason)
        {
            live_lost = true;
            lives_lost_game++;
            lives_lost_section++;
            // 0-water,1-static,2-moving
            if (reason == 0)
            {
                lives_lost_enemy_water_game++;
                lives_lost_enemy_water_section++;
            }
            else if (reason == 1)
            {
                lives_lost_enemy_static_game++;
                lives_lost_enemy_static_section++;
            }
            else if (reason == 2)
            {
                lives_lost_enemy_moving_game++;
                lives_lost_enemy_moving_section++;
            }
        }

        public void CharacterJump()
        {
            jumps_game++;
            jumps_section++;
        }

        private void WriteCsvSection(Block section)
        {
            object[] outputarray = new object[] { SystemInfo.deviceUniqueIdentifier, game_id, section.name, m_Coin.Value, coins_section, coins_easy_section, coins_hard_section, chest_normal_section, chest_hard_section, total_time_section, total_time_gained_section, lives_lost_section, lives_lost_enemy_water_section, lives_lost_enemy_static_section, lives_lost_enemy_moving_section, m_HighScore, m_Score, player_backtracked_section, jumps_section, average_velocity_section };
            StringBuilder sbOutput = new StringBuilder();
            sbOutput.Append(string.Join(";", outputarray));
            this.csv_section.WriteLine(sbOutput.ToString());
        }
        private void WriteCsvGame()
        {
            object[] outputarray = new object[] { SystemInfo.deviceUniqueIdentifier, game_id, m_Coin.Value, coins_game, coins_easy_game, coins_hard_game, chest_normal_game, chest_hard_game, total_time_game, total_time_gained_game, lives_lost_game, lives_lost_enemy_water_game, lives_lost_enemy_static_game, lives_lost_enemy_moving_game, m_HighScore, m_Score, player_backtracked_game, jumps_game, average_velocity_game };
            StringBuilder sbOutput = new StringBuilder();
            sbOutput.Append(string.Join(";", outputarray));
            this.csv_game.WriteLine(sbOutput.ToString());
        }

        IEnumerator DeathCrt(float wait_time)
        {
            m_LastScore = m_Score;
            if (m_Score > m_HighScore)
            {
                m_HighScore = m_Score;
            }
            if (OnScoreChanged != null)
            {
                OnScoreChanged(m_Score, m_HighScore, m_LastScore);
            }

            yield return new WaitForSecondsRealtime(wait_time);

            EndGame(true);
            var endScreen = UIManager.Singleton.UISCREENS.Find(el => el.ScreenInfo == UIScreenInfo.END_SCREEN);
            UIManager.Singleton.OpenScreen(endScreen);
        }

        private void Start()
        {
            m_MainCharacter.IsDead.AddEventAndFire(UpdateDeathEvent, this);
            m_StartScoreX = m_MainCharacter.transform.position.x;
            Init();
        }

        public void Init()
        {
            EndGame(false);
            UIManager.Singleton.Init();
            StartCoroutine(Load());
            m_Timer = max_game_time;
        }

        void Update()
        {
            if (m_GameRunning)
            {
                if (m_MainCharacter.transform.position.x > m_StartScoreX && m_MainCharacter.transform.position.x > m_Score)
                {
                    m_Score = m_MainCharacter.transform.position.x;
                    if (OnScoreChanged != null)
                    {
                        OnScoreChanged(m_Score, m_HighScore, m_LastScore);
                    }
                }
                else if (m_MainCharacter.transform.position.x > m_StartScoreX && m_MainCharacter.transform.position.x + 10 < m_Score && !player_backtracked_section)
                {
                    if (live_lost)
                    {
                        m_Score = m_MainCharacter.transform.position.x;
                        if (OnScoreChanged != null)
                        {
                            OnScoreChanged(m_Score, m_HighScore, m_LastScore);
                        }
                        live_lost = false;
                    }
                    else
                    {
                        player_backtracked_section = true;
                    }
                }
                this.average_velocity_game += ((m_MainCharacter.Rigidbody2D.velocity.x - this.average_velocity_game) / ++this.average_velocity_game_i);
                this.average_velocity_section += ((m_MainCharacter.Rigidbody2D.velocity.x - this.average_velocity_section) / ++this.average_velocity_section_i);
                total_time_game += Time.deltaTime;
                total_time_section += Time.deltaTime;
                var now_current_block_position = TerrainGenerator.Singleton.GetCharacterBlockPositionX();
                if (now_current_block_position != current_block_position && now_current_block_position > current_block_position)
                {
                    if (player_backtracked_section) this.player_backtracked_game++;
                    this.WriteCsvSection(this.current_block);

                    this.coins_section = 0;
                    this.coins_easy_section = 0;
                    this.coins_hard_section = 0;
                    this.chest_normal_section = 0;
                    this.chest_hard_section = 0;
                    this.total_time_section = 0;
                    this.total_time_gained_section = 0;
                    this.lives_lost_section = 0;
                    this.lives_lost_enemy_water_section = 0;
                    this.lives_lost_enemy_static_section = 0;
                    this.lives_lost_enemy_moving_section = 0;
                    this.player_backtracked_section = false;
                    this.jumps_section = 0;
                    this.average_velocity_section = 0;
                    this.average_velocity_section_i = 0;
                    this.current_block_position = now_current_block_position;
                    this.current_block = TerrainGenerator.Singleton.GetCharacterBlock();
                }

                m_Timer -= Time.deltaTime;
                if (m_Timer <= 0)
                {
                    m_Timer = 0;
                    StartCoroutine(DeathCrt(0.1f));
                }
                if (OnTimerChanged != null)
                {
                    OnTimerChanged(m_Timer);
                }
            }
        }

        IEnumerator Load()
        {
            var startScreen = UIManager.Singleton.UISCREENS.Find(el => el.ScreenInfo == UIScreenInfo.START_SCREEN);
            yield return new WaitForSecondsRealtime(3f);
            UIManager.Singleton.OpenScreen(startScreen);
        }

        void OnApplicationQuit()
        {
            if (m_Score > m_HighScore)
            {
                m_HighScore = m_Score;
            }
            if (m_GameStarted)
            {
                if (player_backtracked_section) this.player_backtracked_game++;
                WriteCsvGame();
                WriteCsvSection(this.current_block);
            }
            this.csv_game.Close();
            this.csv_section.Close();
            SaveGame.Save<int>("coin", m_Coin.Value);
            SaveGame.Save<int>("id", game_id);
            SaveGame.Save<float>("lastScore", m_Score);
            SaveGame.Save<float>("highScore", m_HighScore);
        }

        public void ExitGame()
        {
            Application.Quit();
        }

        public void ToggleAudioEnabled()
        {
            SetAudioEnabled(!m_AudioEnabled);
        }

        public void SetAudioEnabled(bool active)
        {
            m_AudioEnabled = active;
            AudioListener.volume = active ? 1f : 0f;
            if (OnAudioEnabled != null)
            {
                OnAudioEnabled(active);
            }
        }

        public void StartGame()
        {
            m_GameStarted = true;
            this.current_block = TerrainGenerator.Singleton.GetCharacterBlock();
            this.current_block_position = TerrainGenerator.Singleton.GetCharacterBlockPositionX();
            this.score_game = 0;
            this.coins_game = 0;
            this.coins_easy_game = 0;
            this.coins_hard_game = 0;
            this.chest_normal_game = 0;
            this.chest_hard_game = 0;
            this.total_time_game = 0;
            this.total_time_gained_game = 0;
            this.lives_lost_game = 0;
            this.lives_lost_enemy_water_game = 0;
            this.lives_lost_enemy_static_game = 0;
            this.lives_lost_enemy_moving_game = 0;
            this.player_backtracked_game = 0;
            this.jumps_game = 0;
            this.average_velocity_game = 0;
            this.average_velocity_game_i = 0;

            this.score_section = 0;
            this.coins_section = 0;
            this.coins_easy_section = 0;
            this.coins_hard_section = 0;
            this.chest_normal_section = 0;
            this.chest_hard_section = 0;
            this.total_time_section = 0;
            this.total_time_gained_section = 0;
            this.lives_lost_section = 0;
            this.lives_lost_enemy_water_section = 0;
            this.lives_lost_enemy_static_section = 0;
            this.lives_lost_enemy_moving_section = 0;
            this.player_backtracked_section = false;
            this.jumps_section = 0;
            this.average_velocity_section = 0;
            this.average_velocity_section_i = 0;
            live_lost = false;
            ResumeGame();
        }

        public void StopGame()
        {
            m_GameRunning = false;
            Time.timeScale = 0f;
        }

        public void ResumeGame()
        {
            m_GameRunning = true;
            Time.timeScale = 1f;
        }

        public void EndGame(bool save)
        {
            if (save)
            {
                if (player_backtracked_section) this.player_backtracked_game++;
                WriteCsvSection(this.current_block);
                WriteCsvGame();
            }
            m_GameStarted = false;
            StopGame();
        }

        public void Reset()
        {
            m_Score = 0f;
            m_Timer = max_game_time;
            if (OnReset != null)
            {
                OnReset();
            }
        }

        public void ShareOnTwitter()
        {
            Share("https://twitter.com/intent/tweet?text={0}&url={1}");
        }

        public void ShareOnGooglePlus()
        {
            Share("https://plus.google.com/share?text={0}&href={1}");
        }

        public void ShareOnFacebook()
        {
            Share("https://www.facebook.com/sharer/sharer.php?u={1}");
        }

        public void Share(string url)
        {
            Application.OpenURL(string.Format(url, m_ShareText, m_ShareUrl));
        }

        [System.Serializable]
        public class LoadEvent : UnityEvent
        {

        }

    }

}