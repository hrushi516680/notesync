from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def send_whatsapp_summary(message_to_send, group_name="project demo"):
    edge_options = Options()
    driver = webdriver.Edge(options=edge_options)
    driver.get("https://web.whatsapp.com/")

    wait = WebDriverWait(driver, 60)

    try:
        print("📲 Waiting for WhatsApp Web to load...")
        wait.until(EC.presence_of_element_located((By.ID, "side")))
        print("✅ WhatsApp Web loaded!")

        # 💥 Wait until annoying spinner disappears
        print("🟡 Waiting for the green popup spinner to vanish...")
        try:
            WebDriverWait(driver, 15).until_not(
                EC.presence_of_element_located((By.XPATH, '//span[@data-icon="ui-refresh-nux-bg"]'))
            )
            print("✅ Spinner gone. Safe to proceed.")
        except:
            print("⚠️ Spinner not found or already gone.")

        # ✅ Now search for the group
        print(f"🔎 Trying to search for group: {group_name}")
        try:
            search_box = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//div[@contenteditable="true"][@data-tab="3"]')))
            search_box.click()
            search_box.send_keys(group_name)
            time.sleep(2)
        except Exception as e:
            print(f"❌ Failed to click search box: {e}")
            driver.quit()
            return

        # ✅ Click the group chat
        try:
            chat = wait.until(EC.element_to_be_clickable(
                (By.XPATH, f'//span[@title="{group_name}"]')))
            chat.click()
            print(f"📂 Group '{group_name}' selected.")
        except Exception as e:
            print(f"❌ Failed to select group '{group_name}': {e}")
            driver.quit()
            return

        # ✅ Type and send message
        try:
            message_box = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '//div[@contenteditable="true"][@data-tab="1"]')))
            message_box.click()
            message_box.send_keys(message_to_send)
            message_box.send_keys(Keys.ENTER)
            print("✅ Message sent successfully!")
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
            driver.quit()
            return

    except Exception as e:
        print(f"💥 Unexpected Error: {e}")

    finally:
        time.sleep(2)
        driver.quit()
        print("🛑 Browser closed.")
